#include <iostream>
#include <vector>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "hip_utils.hpp"
#include "kernels.hpp"

auto main() -> int
{
    try 
    {
        std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> Y = {0.0f, 0.0f};
        
        float a = 1.0f, b = 1.0f;
        const int m = static_cast<int>(Y.size());
        const int n = static_cast<int>(x.size());

        thrust::device_vector<float> d_A = A;
        thrust::device_vector<float> d_x = x;
        thrust::device_vector<float> d_Y = Y;

        constexpr int block = 256;
        const int grid = (m + block - 1) / block;

        
        gemv_kernel<<<grid, block>>>(
            a, 
            thrust::raw_pointer_cast(d_A.data()), 
            thrust::raw_pointer_cast(d_x.data()), 
            b, 
            thrust::raw_pointer_cast(d_Y.data()), 
            m, n
        );
        
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        thrust::copy(d_Y.begin(), d_Y.end(), Y.begin());

        for (int i = 0; i < m; ++i)
        {
            std::cout << "Y[" << i << "] = " << Y[i] << '\n';
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Erreur Thrust/C++ : " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return 0;
}