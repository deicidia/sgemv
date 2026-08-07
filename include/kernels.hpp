#pragma once

#include <hip/hip_runtime.h>

__global__ void sgemv_naive(float a, const float* A, const float* x, float b, float* Y, int m, int n);

__global__ void sgemv_block(float a, const float* A, const float* x, float b, float* Y, int m, int n);

__global__ void sgemv_wave32(float a, const float* A, const float* x, float b, float* Y, int m, int n);

__global__ void sgemv_wave32_float4(float a, const float* A, const float* x, float b, float* Y, int m, int n);
