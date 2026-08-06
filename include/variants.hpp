#pragma once

#include <string>
#include <vector>

#include "kernels.hpp"

// Every variant shares this signature, so a plain function pointer is enough —
// no launcher function per variant, no std::function.
using KernelPtr = void (*)(float, const float*, const float*, float, float*, int, int);

struct Variant
{
    std::string name;
    KernelPtr   kernel;
    int         threads_per_row;   // 1 = one thread, 32 = one wave, 256 = one block
};

// Single source of truth for both the benchmark and the test binary. The launch
// geometry is derived from threads_per_row rather than written out per variant,
// which is what stops the two from drifting apart.
inline const std::vector<Variant>& variants()
{
    static const std::vector<Variant> v = {
        {"naive",  sgemv_naive,    1},
        {"block",  sgemv_block,  256},
        {"wave32", sgemv_wave32,  32},
    };
    return v;
}

constexpr int kBlockSize = 256;

inline int grid_for(const Variant& v, int m)
{
    return (m * v.threads_per_row + kBlockSize - 1) / kBlockSize;
}

inline const Variant* find_variant(const std::string& name)
{
    for (const auto& v : variants())
    {
        if (v.name == name) return &v;
    }
    return nullptr;
}
