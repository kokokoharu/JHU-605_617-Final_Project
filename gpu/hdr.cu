#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "hdr.h"

__global__ void computeWeightGpu(
    float * const im,
    float * weight,
    int width,
    int height,
    int channels,
    float epsilonMini,
    float epsilonMaxi,
    bool isFirst,
    bool isLast
)
{
    // Each thread should work on one pixel in each channel and across all channels
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < width && out_y < height)
    {
        for (int z = 0; z < channels; ++z)
        {
            float pixelVal = im_smartAccessClamp(im, width, height, channels, out_x, out_y, z);
            float weightVal = 0.0f;

            if (isFirst)
            {
                weightVal = (pixelVal >= epsilonMini) ? 1.0f : 0.0f;
            }
            else if (isLast)
            {
                weightVal = (pixelVal <= epsilonMaxi) ? 1.0f : 0.0f;
            }
            else
            {
                weightVal = (pixelVal >= epsilonMini && pixelVal <= epsilonMaxi) ? 1.0f : 0.0f;
            }

            im_smartSetterClamp(weight, width, height, channels, out_x, out_y, z, weightVal);
        }
    }

    return;
}

__global__ void hdrMergeGpu(
    float ** imSeq_d_arr,
    float ** weightMaps_d_arr,
    float * factors_d,
    float * hdr_d,
    int width,
    int height,
    int channels,
    int num_images
)
{
    // Each thread is responsible for one output pixel across all channels
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < width && out_y < height)
    {
        for (int k = 0; k < channels; ++k)
        {
            float weight_accum = 0.0f;
            float pixel_tmp = 0.0f;

            for (int im = 0; im < num_images; ++im)
            {
                float weight_tmp = im_smartAccessClamp(weightMaps_d_arr[im], width, height, channels, out_x, out_y, k);
                weight_accum += weight_tmp;
                pixel_tmp += weight_tmp * (1/factors_d[im]) * im_smartAccessClamp(imSeq_d_arr[im], width, height, channels, out_x, out_y, k);
            }

            if (weight_accum == 0.0f)
            {
                float first = im_smartAccessClamp(imSeq_d_arr[0], width, height, channels, out_x, out_y, k);
                im_smartSetterClamp(hdr_d, width, height, channels, out_x, out_y, k, first);
            }
            else
            {
                im_smartSetterClamp(hdr_d, width, height, channels, out_x, out_y, k, pixel_tmp/weight_accum);
            }
        }
    }

    return;
}

Image makeHdrGpuBasic(vector<Image> &imSeq, float epsilonMini, float epsilonMaxi)
{
    if (imSeq.size() == 1)
    {
        cout << "provide more than one image" << endl;
        return imSeq[0];
    }

    // Host definitions
    int N = imSeq.size();
    int width = imSeq[0].width();
    int height = imSeq[0].height();
    int channels = imSeq[0].channels();
    size_t im_num_pixel = imSeq[0].number_of_elements();
    Image hdr(width, height, channels);
    vector<float> factors_h(N);

    // CUDA definitions
    unsigned int threadsPerBlock_x = 16;
    unsigned int threadsPerBlock_y = 16;
    unsigned int blocksPerGrid_x = (width + threadsPerBlock_x - 1) / threadsPerBlock_x;
    unsigned int blocksPerGrid_y = (height + threadsPerBlock_y - 1) / threadsPerBlock_y;
    dim3 threadsPerBlock(threadsPerBlock_x, threadsPerBlock_y);
    dim3 blocksPerGrid(blocksPerGrid_x, blocksPerGrid_y);

    // Pointers to device memory
    vector<float*> imSeq_d(N, nullptr);         // Holds pointers to images in device memory. On host.
    vector<float*> weightMaps_d(N, nullptr);    // Holds pointers to weight maps in device memory. On host.
    float ** imSeq_d_ptr = nullptr;             // Holds pointers to images in device memory. On device.
    float ** weightMaps_d_ptr = nullptr;        // Holds pointers to weight maps in device memory. On device.
    float * factors_d = nullptr;
    float * hdr_d = nullptr;

    // Allocate device memory for images and weight maps. Then upload all images.
    for (int i = 0; i < N; ++i)
    {
        CUDA_CHECK(cudaMalloc((void **)&imSeq_d[i], im_num_pixel * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&weightMaps_d[i], im_num_pixel * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(imSeq_d[i], &imSeq[i](0), im_num_pixel * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Compute weights on device
    for (int i = 0; i < N; ++i)
    {
        bool isFirst = i == 0;
        bool isLast = i == (N - 1);

        computeWeightGpu<<<blocksPerGrid, threadsPerBlock>>>(
            imSeq_d[i],
            weightMaps_d[i],
            width,
            height,
            channels,
            epsilonMini,
            epsilonMaxi,
            isFirst,
            isLast
        );
    }

    CUDA_CHECK_KERNEL();

    // Download weights and compute relative exposure factors on host
    // Set up for the 1st image
    factors_h[0] = 1.0f;
    Image w_prev(width, height, channels);
    CUDA_CHECK(cudaMemcpy(&w_prev(0), weightMaps_d[0], im_num_pixel * sizeof(float), cudaMemcpyDeviceToHost));
    float current_ratio;
    for (int i = 1; i < N; ++i)
    {
        Image w_curr(width, height, channels);
        CUDA_CHECK(cudaMemcpy(&w_curr(0), weightMaps_d[i], im_num_pixel * sizeof(float), cudaMemcpyDeviceToHost));
        current_ratio = computeFactor(imSeq[i-1], w_prev, imSeq[i], w_curr);
        factors_h[i] = current_ratio * factors_h[i-1];
        w_prev = w_curr;
    }

    // Upload relative exposure factors
    CUDA_CHECK(cudaMalloc((void **)&factors_d, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(factors_d, factors_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Perform HDR merge on device
    // Upload array of device image pointers and array of device weight pointers
    CUDA_CHECK(cudaMalloc((void **)&imSeq_d_ptr, N * sizeof(float*)));
    CUDA_CHECK(cudaMalloc((void **)&weightMaps_d_ptr, N * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(imSeq_d_ptr, imSeq_d.data(), N * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weightMaps_d_ptr, weightMaps_d.data(), N * sizeof(float*), cudaMemcpyHostToDevice));

    // Allocate device memory for output HDR image and launch kernel
    CUDA_CHECK(cudaMalloc((void **)&hdr_d, im_num_pixel * sizeof(float)));
    CUDA_CHECK(cudaMemset(hdr_d, 0, im_num_pixel * sizeof(float)));
    hdrMergeGpu<<<blocksPerGrid, threadsPerBlock>>>(
        imSeq_d_ptr,
        weightMaps_d_ptr,
        factors_d,
        hdr_d,
        width,
        height,
        channels,
        N
    );

    CUDA_CHECK_KERNEL();

    // Download the HDR image
    CUDA_CHECK(cudaMemcpy(&hdr(0), hdr_d, im_num_pixel * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    for (int i = 0; i < N; ++i)
    {
        cudaFree(imSeq_d[i]);
        cudaFree(weightMaps_d[i]);
    }
    cudaFree(imSeq_d_ptr);
    cudaFree(weightMaps_d_ptr);
    cudaFree(factors_d);
    cudaFree(hdr_d);
    
    return hdr;
}