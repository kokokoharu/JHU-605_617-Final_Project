#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "hdr.h"

__global__ void computeWeightGpu(float * im, )

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
    int im_num_pixel = width * height * channels;

    // Pointers to device memory
    vector<float*> imSeq_d(N);
    vector<float*> weightMaps_d(N);

    // Allocate device memory for images and weight maps. Then upload all images.
    for (int i = 0; i < N; ++i)
    {
        CUDA_CHECK(cudaMalloc((void **)&imSeq_d[i], im_num_pixel * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&weightMaps_d[i], im_num_pixel * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(imSeq_d[i], &imSeq[i](0), im_num_pixel * sizeof(float)));
    }

    // TODO: Compute weights on device (compute weight kernel)
    // TODO: Download weights and compute factors on host
    // TODO: Upload factors
    // TODO: Perform HDR merge on device
}