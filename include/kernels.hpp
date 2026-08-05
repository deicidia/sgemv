#pragma once

#include <hip/hip_runtime.h>

__global__ void gemv_kernel(float a, const float* A, const float* x, float b, float* Y, int m, int n);