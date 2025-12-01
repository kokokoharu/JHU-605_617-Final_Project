#include <cuda_runtime.h>
#include <iostream>
#include "cuda_utils.h"
#include "filtering.h"

using namespace std;

// Hard-code the following values for tiled bilateral filtering
#define BILA_TILE_DIM                           (32)
#define BILA_TILE_SIG_RANGE                     (0.1f)
#define BILA_TILE_SIG_DOMAIN                    (1.0f)
#define BILA_TILE_TRUNC_DOMAIN                  (3.0f)

// Switch between basic bilateral kernel and tiled bilateral kernel
#define USE_TILE_BILA                           (1)

// A tiled bilateral filtering kernel with L2 caching of the halo cells.
// We access halo cells from global memory and hope they are cached in L2, especially for those halo cells internal to other tiles.
__global__ void bilateral_tiled_with_l2_cache(
    float * const im_in,
    float * im_out,
    int width,
    int height,
    int channels
)
{
    float accum, normalizer, range_dist, tmp, factorDomain, factorRange;
    int filter_radius = int(ceil(BILA_TILE_TRUNC_DOMAIN * BILA_TILE_SIG_DOMAIN));

    // Each thread should work on one pixel in each channel and across all channels
    int out_x = blockIdx.x * BILA_TILE_DIM + threadIdx.x;
    int out_y = blockIdx.y * BILA_TILE_DIM + threadIdx.y;

    // Load input tile with clamping
    // IMPORTANT: Image cannot be of more than 3 channels
    __shared__ float tile_in[3][BILA_TILE_DIM][BILA_TILE_DIM];
    for (int z = 0; z < channels; ++z)
    {
        tile_in[z][threadIdx.y][threadIdx.x] = im_smartAccessClamp(im_in, width, height, channels, out_x, out_y, z);
    }

    __syncthreads();

    // Calculate output elements
    if (out_x < width && out_y < height)
    {
        for (int z = 0; z < channels; ++z)
        {
            normalizer = 0.0f;
            accum = 0.0f;

            // Loop over filter's support
            for (int yFilter = 0; yFilter < 2*filter_radius+1; ++yFilter)
            {
                for (int xFilter = 0; xFilter < 2*filter_radius+1; ++xFilter)
                {
                    // Calculate the distance between the 2 pixels in range
                    range_dist = 0.0f; // |R-R1|^2 + |G-G1|^2 + |B-B1|^2
                    for (int z1 = 0; z1 < channels; ++z1)
                    {
                        // Input center pixel, must be inside input tile
                        tmp = tile_in[z1][threadIdx.y][threadIdx.x];

                        // Find distance with a neighbor
                        // If neighbor is inside input tile, get its value from shared memory
                        if (threadIdx.x - filter_radius + xFilter >= 0
                            && threadIdx.x - filter_radius + xFilter < BILA_TILE_DIM
                            && threadIdx.y - filter_radius + yFilter >= 0
                            && threadIdx.y - filter_radius + yFilter < BILA_TILE_DIM)
                        {
                            tmp -= tile_in[z1][threadIdx.y-filter_radius+yFilter][threadIdx.x-filter_radius+xFilter];
                        }
                        // If not, get its value from global memory and hope that the value has been cached in L2
                        else
                        {
                            tmp -= im_smartAccessClamp(
                                im_in,
                                width, height, channels,
                                out_x - filter_radius + xFilter,
                                out_y - filter_radius + yFilter,
                                z1);
                        }

                        // Square
                        tmp *= tmp;

                        range_dist += tmp;
                    }

                    // calculate the exponential weight from the domain and range
                    factorDomain = exp( - ((xFilter-filter_radius)*(xFilter-filter_radius) + (yFilter-filter_radius)*(yFilter-filter_radius) ) / (2.0 * BILA_TILE_SIG_DOMAIN*BILA_TILE_SIG_DOMAIN ) );
                    factorRange  = exp( - range_dist / (2.0 * BILA_TILE_SIG_RANGE*BILA_TILE_SIG_RANGE) );

                    normalizer += factorDomain * factorRange;

                    if (threadIdx.x - filter_radius + xFilter >= 0
                            && threadIdx.x - filter_radius + xFilter < BILA_TILE_DIM
                            && threadIdx.y - filter_radius + yFilter >= 0
                            && threadIdx.y - filter_radius + yFilter < BILA_TILE_DIM)
                    {
                        accum += factorDomain * factorRange * tile_in[z][threadIdx.y-filter_radius+yFilter][threadIdx.x-filter_radius+xFilter];
                    }
                    else
                    {
                        accum += factorDomain * factorRange * im_smartAccessClamp(
                            im_in,
                            width, height, channels,
                            out_x - filter_radius + xFilter,
                            out_y - filter_radius + yFilter,
                            z);
                    }
                }
            }

            im_out[z*width*height + out_y*width + out_x] = accum / normalizer;
        }
    }
    return;
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

    unsigned int threadsPerBlock_x;
    unsigned int threadsPerBlock_y;
    if (USE_TILE_BILA == 1)
    {
        threadsPerBlock_x = BILA_TILE_DIM;
        threadsPerBlock_y = BILA_TILE_DIM;
    }
    else
    {
        threadsPerBlock_x = 16;
        threadsPerBlock_y = 16;
    }
    unsigned int blocksPerGrid_x = (im.width() + threadsPerBlock_x - 1) / threadsPerBlock_x;
    unsigned int blocksPerGrid_y = (im.height() + threadsPerBlock_y - 1) / threadsPerBlock_y;
    dim3 threadsPerBlock(threadsPerBlock_x, threadsPerBlock_y);
    dim3 blocksPerGrid(blocksPerGrid_x, blocksPerGrid_y);
    size_t im_num_pixel = im.number_of_elements();

    // Pointers to device memory
    float * im_d = nullptr;
    float * im_out_d = nullptr;

    // Device memory allocation
    CUDA_CHECK(cudaMalloc((void **)&im_d, im_num_pixel * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&im_out_d, im_num_pixel * sizeof(float)));

    // H->D
    CUDA_CHECK(cudaMemcpy(im_d, &im(0), im_num_pixel * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    if (USE_TILE_BILA == 1)
    {
        printf("Enqueue (tiled) bilateral kernel with gridDim (%d,%d,%d) blockDim (%d,%d,%d) filterRadius %d sigmaRange %f sigmaDomain %f truncateDomain %f\n",
            blocksPerGrid.z,
            blocksPerGrid.y,
            blocksPerGrid.x,
            threadsPerBlock.z,
            threadsPerBlock.y,
            threadsPerBlock.x,
            int(ceil(BILA_TILE_TRUNC_DOMAIN * BILA_TILE_SIG_DOMAIN)),
            BILA_TILE_SIG_RANGE,
            BILA_TILE_SIG_DOMAIN,
            BILA_TILE_TRUNC_DOMAIN);

        bilateral_tiled_with_l2_cache<<<blocksPerGrid, threadsPerBlock>>>(
            im_d,
            im_out_d,
            im.width(),
            im.height(),
            im.channels());
    }
    else
    {
        printf("Enqueue (basic) bilateral kernel with gridDim (%d,%d,%d) blockDim (%d,%d,%d) filterRadius %d sigmaRange %f sigmaDomain %f truncateDomain %f\n",
            blocksPerGrid.z,
            blocksPerGrid.y,
            blocksPerGrid.x,
            threadsPerBlock.z,
            threadsPerBlock.y,
            threadsPerBlock.x,
            int(ceil(truncateDomain * sigmaDomain)),
            sigmaRange,
            sigmaDomain,
            truncateDomain);

        bilateral_basic<<<blocksPerGrid, threadsPerBlock>>>(
            im_d,
            im_out_d,
            im.width(),
            im.height(),
            im.channels(),
            sigmaRange,
            sigmaDomain,
            truncateDomain);
    }

    CUDA_CHECK_KERNEL();

    // D->H
    CUDA_CHECK(cudaMemcpy(&imFilter(0), im_out_d, im_num_pixel * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(im_d);
    cudaFree(im_out_d);

    return imFilter;
}
