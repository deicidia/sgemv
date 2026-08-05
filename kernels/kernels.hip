#include "kernels.hpp"

//naive implementation of the GEMV kernel
__global__ void gemv_kernel(float a, const float* A, const float* x, float b, float* Y, int m, int n)
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