#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// For kernel launches (different error checking pattern)
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel execution error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device-side image accessor and setter functions
// These are inline to avoid multiple definition errors when included in multiple .cu files

__device__ inline float im_smartAccessClamp(
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

__device__ inline float im_smartAccessNoClamp(
    float * const im_data,
    int width,
    int height,
    int channels,
    int x,
    int y,
    int z)
{
    // This function never clamps pixel indices, i.e. when a pixel is out of bound we return a black value
    if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= channels)
    {
        return 0.0f;
    }

    return im_data[z*width*height + y*width + x];
}

__device__ inline void im_smartSetterClamp(
    float * im_data,
    int width,
    int height,
    int channels,
    int x,
    int y,
    int z,
    float newVal)
{
    // Image data is stored as z*(width*height) + y*width + x
    // This function always clamps pixel indices, i.e. when a pixel is out of bound we set the nearest valid pixel
    x = max(min(x, width - 1), 0);
    y = max(min(y, height - 1), 0);
    z = max(min(z, channels - 1), 0);

    im_data[z*width*height + y*width + x] = newVal;
}

__device__ inline void im_smartSetterNoClamp(
    float * im_data,
    int width,
    int height,
    int channels,
    int x,
    int y,
    int z,
    float newVal)
{
    // This functions doesn't clamp pixel indices.
    // When a pixel is out of bound, nothing is done.
    if (x >= 0 && y >= 0 && z >= 0 && x < width && y < height && z < channels)
    {
        im_data[z*width*height + y*width + x] = newVal;
    }
}

#endif // CUDA_UTILS_H
