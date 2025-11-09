#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "filtering.h"

__device__ float im_smartAccessClamp(
    float * const im_data,
    int width,
    int height,
    int channels,
    int x,
    int y,
    int z)
{
    // Image data is stored as z*(width*height) + y*width + x
    // This function always clamps pixel indices, i.e. when a pixel is out of bound we use the nearest valid pixel

    x = max(min(x, width - 1), 0);
    y = max(min(y, height - 1), 0);
    z = max(min(z, channels - 1), 0);

    return im_data[z*width*height + y*width + x];
}

__device__ float im_smartAccessNoClamp(
    float * const im_data,
    int width,
    int height,
    int channels,
    int x,
    int y,
    int z)
{
    // This function never clamps pixel indices, i.e. when a pixel is out of bound we return a black value

    if ( x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= channels)
    {
        return 0.0f;
    }

    return im_data[z*width*height + y*width + x];
}

__global__ void bilateral_basic(
    float * const im_in,
    float * im_out,
    int width,
    int height,
    int channels,
    float sigmaRange,
    float sigmaDomain,
    float truncateDomain)
{
    float accum, normalizer, range_dist, tmp, factorDomain, factorRange;
    int offset = int(ceil(truncateDomain * sigmaDomain));
    int sizeFilt = 2 * offset + 1;

    // Each thread should work on one pixel in each channel and across all channels
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < width && out_y < height)
    {
        // Loop over all channels
        for (int z = 0; z < channels; ++z)
        {
            normalizer = 0.0f;
            accum = 0.0f;

            // Loop over filter's support
            for (int yFilter = 0; yFilter < sizeFilt; ++yFilter)
            {
                for (int xFilter = 0; xFilter < sizeFilt; ++xFilter)
                {
                    // Calculate the distance between the 2 pixels in range
                    range_dist = 0.0f; // |R-R1|^2 + |G-G1|^2 + |B-B1|^2
                    for (int z1 = 0; z1 < channels; ++z1)
                    {
                        // Input center pixel
                        tmp = im_smartAccessClamp(
                            im_in,
                            width,
                            height,
                            channels,
                            out_x,
                            out_y,
                            z1);

                        // Find distance with a neighbor
                        tmp -= im_smartAccessClamp(
                            im_in,
                            width,
                            height,
                            channels,
                            out_x + xFilter - offset,
                            out_y + yFilter - offset,
                            z1);
                        
                        // Square
                        tmp *= tmp;
                        
                        range_dist += tmp;
                    }

                    // calculate the exponential weight from the domain and range
                    factorDomain = exp( - ((xFilter-offset)*(xFilter-offset) + (yFilter-offset)*(yFilter-offset) ) / (2.0 * sigmaDomain*sigmaDomain ) );
                    factorRange  = exp( - range_dist / (2.0 * sigmaRange*sigmaRange) );

                    normalizer += factorDomain * factorRange;
                    accum += factorDomain * factorRange * im_smartAccessClamp(
                        im_in,
                        width,
                        height,
                        channels,
                        out_x + xFilter - offset,
                        out_y + yFilter - offset,
                        z);
                }
            }

            im_out[z*width*height + out_y*width + out_x] = accum / normalizer;
        }

    }
    return;
}

Image bilateralGpuBasic(const Image & im, float sigmaRange, float sigmaDomain, float truncateDomain)
{
    // Output image
    Image imFilter(im.width(), im.height(), im.channels());

    unsigned int threadsPerBlock_x = 16;
    unsigned int threadsPerBlock_y = 16;
    unsigned int blocksPerGrid_x = (im.width() + threadsPerBlock_x - 1) / threadsPerBlock_x;
    unsigned int blocksPerGrid_y = (im.height() + threadsPerBlock_y - 1) / threadsPerBlock_y;
    dim3 threadsPerBlock(threadsPerBlock_x, threadsPerBlock_y);
    dim3 blocksPerGrid(blocksPerGrid_x, blocksPerGrid_y);
    size_t im_num_pixel = im.number_of_elements();

    // Device definition
    float * im_d = nullptr;
    float * im_out_d = nullptr;

    // Device memory allocation
    CUDA_CHECK(cudaMalloc((void **)&im_d, im_num_pixel * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&im_out_d, im_num_pixel * sizeof(float)));

    // H->D
    CUDA_CHECK(cudaMemcpy(im_d, &im(0), im_num_pixel * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    bilateral_basic<<<blocksPerGrid, threadsPerBlock>>>(
        im_d,
        im_out_d,
        im.width(),
        im.height(),
        im.channels(),
        sigmaRange,
        sigmaDomain,
        truncateDomain);

    CUDA_CHECK_KERNEL();

    // D->H
    CUDA_CHECK(cudaMemcpy(&imFilter(0), im_out_d, im_num_pixel * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(im_d);
    cudaFree(im_out_d);

    return imFilter;
}
