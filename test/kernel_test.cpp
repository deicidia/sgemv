#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "hip_utils.hpp"
#include "variants.hpp"

struct Case 
{
    std::string name; 
    int m;
    int n;
    float a;
    float b;
};

// Use double precision for the reference implementation to reduce numerical errors
auto reference(const std::vector<float>& A, const std::vector<float>& x,
               const std::vector<float>& Y_init, float a, float b,
               int m, int n) -> std::vector<double>
{
    std::vector<double> Y(m);

    for (int i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
        {
            sum += static_cast<double>(A[i * n + j]) * static_cast<double>(x[j]);
        }
        Y[i] = static_cast<double>(a) * sum
             + static_cast<double>(b) * static_cast<double>(Y_init[i]);
    }

    return Y;
}

auto run_kernel(const Variant& v,
                const std::vector<float>& A, const std::vector<float>& x,
                const std::vector<float>& Y_init, float a, float b,
                int m, int n) -> std::vector<float>
{

    thrust::device_vector<float> d_A = A;
    thrust::device_vector<float> d_x = x;
    thrust::device_vector<float> d_Y = Y_init;

    // Same geometry rule as the benchmark, from the same table.
    const int grid = grid_for(v, m);

    v.kernel<<<grid, kBlockSize>>>(
        a,
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_x.data()),
        b,
        thrust::raw_pointer_cast(d_Y.data()),
        m, n
    );

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> Y(m);
    thrust::copy(d_Y.begin(), d_Y.end(), Y.begin());

    return Y;
}

// Tolerance function based on the row size of the matrix and machine epsilon for float
auto tol(int n) -> double
{
    return std::numeric_limits<float>::epsilon() * (4.0 * n + 8.0);
}

auto run_case(const Variant& v, const Case& c) -> bool
{
    std::vector<float> A(c.m * c.n);
    std::vector<float> x(c.n);
    std::vector<float> Y_init(c.m);

    unsigned int seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
    std::generate(x.begin(), x.end(), [&]() { return dis(gen); });
    std::generate(Y_init.begin(), Y_init.end(), [&]() { return dis(gen); });

    auto Y_ref = reference(A, x, Y_init, c.a, c.b, c.m, c.n);
    auto Y_test = run_kernel(v, A, x, Y_init, c.a, c.b, c.m, c.n);

    bool success = true;
    double max_err = 0.0;

    for (int i = 0; i < c.m; ++i)
    {
        const double err = std::abs(static_cast<double>(Y_ref[i]) - static_cast<double>(Y_test[i]));
        max_err = std::max(max_err, err);
        if (err > tol(c.n)) success = false;
    }

    std::cout << std::left << std::setw(24) << c.name
        << " m=" << std::setw(6) << c.m
        << " n=" << std::setw(6) << c.n
        << std::scientific << std::setprecision(2)
        << " tol=" << std::setw(10) << tol(c.n)
        << " err=" << std::setw(10) << max_err
        << (success ? "  PASS" : "  FAIL") << '\n';

    return success;
}

auto main() -> int
{
    std::vector<Case> cases = {
        {"square",                  64,   64,  1.0f,  0.0f},
        {"unaligned dims",         257,   33,  2.0f,  3.0f},
        {"tall & thin (decode)",  4096,   16,  1.5f, -0.5f},
        {"short & wide",            16, 4096,  1.0f,  1.0f},
        {"negative alpha",        1024, 1024, -1.0f,  2.0f},
    };

    try
    {
        bool all_passed = true;
        // Every variant against the same double-precision reference: a correct
        // kernel lands on it whatever its geometry or reduction order.
        for(auto const & v : variants())
        {
            std::cout << "=== " << v.name << " ===\n";
            for(auto const & c : cases)
            {
                if(!run_case(v, c)) all_passed = false;
            }
            std::cout << '\n';
        }
        std::cout << (all_passed ? "All test cases passed." : "Some test cases failed.") << std::endl;
        return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Erreur Thrust/C++ : " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}