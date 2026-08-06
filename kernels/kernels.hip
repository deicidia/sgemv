#include "kernels.hpp"

// Naive implementation of the GEMV kernel
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

// Block implementation of the GEMV kernel
__global__ void sgemv_block(float a, const float* A, const float* x, float b, float* Y, int m, int n)
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

// Warp implementation of the GEMV kernel
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