#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                        \
    do                                                                        \
    {                                                                         \
        hipError_t err = (cmd);                                               \
        if (err != hipSuccess)                                                \
        {                                                                     \
            std::fprintf(stderr, "%s:%d  %s -> %s\n", __FILE__, __LINE__,     \
                         #cmd, hipGetErrorString(err));                       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)


#define GRID_STRIDE_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
