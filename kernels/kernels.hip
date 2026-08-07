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

// Warp implementation of the sgemv kernel 
__global__ void sgemv_wave32(float a, const float* A, const float* x, float b, float* Y, int m, int n)
{
    const int lane  = threadIdx.x % warpSize; // lane index in the warp
    const int wave  = threadIdx.x / warpSize; // wave index in the block
    const int waves = blockDim.x / warpSize;  // number of waves in the block
    const int row   = blockIdx.x * waves + wave; // row index in the matrix

   if (row >= m) return;

    float local_sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n; i += warpSize)
    {
        local_sum += A[static_cast<size_t>(row) * n + i] * x[i];
    }

    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off, warpSize);

    if (lane == 0) Y[row] = a * local_sum + b * Y[row];
}

// Wrap implementation of the sgemv kernel using float4 
__global__ void sgemv_wave32_float4(float a, const float* A, const float* x, float b, float* Y, int m, int n)
{
    const int lane  = threadIdx.x % warpSize; // lane index in the warp
    const int wave  = threadIdx.x / warpSize; // wave index in the block
    const int waves = blockDim.x / warpSize;  // number of waves in the block
    const int row   = blockIdx.x * waves + wave; // row index in the matrix

    if (row >= m) return;

    float local_sum = 0.0f;

    if(n % 4 == 0)
    {
        int n4 = n / 4;
        // Load A and x as float4 for coalesced memory access
        const float4* A4 = reinterpret_cast<const float4*>(A + static_cast<size_t>(row) * n);
        const float4* x4 = reinterpret_cast<const float4*>(x);

        #pragma unroll 4
        for (int i = lane; i < n4; i += warpSize)
        {
            const float4 av = A4[i];
            const float4 xv = x4[i];
            
            // Unpack the float4 and compute the dot product
            local_sum += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
        }
    }
    else
    {
        #pragma unroll 4
        for (int i = lane; i < n; i += warpSize)
        {
            local_sum += A[static_cast<size_t>(row) * n + i] * x[i];
        }
    }

    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off, warpSize);

    if (lane == 0) Y[row] = a * local_sum + b * Y[row];
}