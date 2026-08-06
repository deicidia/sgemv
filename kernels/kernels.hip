#include "kernels.hpp"
#include "hip_utils.hpp"

// Naive implementation of the sgemv kernel, each thread computes one row of the output vector
__global__ void sgemv_naive(float a, const float* A, const float* x, float b, float* Y, int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col)
        {
            sum += A[row * n + col] * x[col];
        }
        Y[row] = a * sum + b * Y[row];
    }
}

/**  
 * Block implementation of the sgemv kernel, each block computes one row of the output vector using shared memory for reduction.
 * Inspired by Mark Harris https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf for the reduction part.
 * To much synchronization overhead, not efficient for small matrices, which lead to the warp implementation below.
 * */ 
__global__ void sgemv_block(float a, const float* A, const float* x, float b, float* Y, int m, int n)
{
    __shared__ float shared_sum[256]; 
    const int row = blockIdx.x;
    float local_sum = 0.0f;

    // One block per row of the matrix
    for(int i = threadIdx.x; i < n; i += blockDim.x)
    {
        local_sum += A[row * n + i] * x[i];
    }

    shared_sum[threadIdx.x] = local_sum;

    __syncthreads();

    // Reduction within the block
    for (int stride = blockDim.x / 2; stride >= warpSize; stride >>= 1)
    {
        if (threadIdx.x < stride) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        __syncthreads();
    }
    
    // Reduction within the warp
    float v = shared_sum[threadIdx.x];
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v += __shfl_xor(v, off, warpSize);

    if (threadIdx.x == 0) Y[row] = a * v + b * Y[row];
}

// Warp implementation of the sgemv kernel : TODO 
__global__ void sgemv_wave32(float a, const float* A, const float* x, float b, float* Y, int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col)
        {
            sum += A[row * n + col] * x[col];
        }
        Y[row] = a * sum + b * Y[row];
    }
}