#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "hip_utils.hpp"
#include "variants.hpp"

namespace
{

constexpr int    kWarmup   = 5;    // absorbs code-object load and clock ramp-up
constexpr int    kIters    = 50;
constexpr double kPeakBps  = 613e9;  // BabelStream 5.0, Copy, float, 403 MB

struct Result
{
    double ms;
    double bw;
    double efficiency;
};

auto benchmark(const Variant& v, int m, int n,
               const thrust::device_vector<float>& d_A,
               const thrust::device_vector<float>& d_x,
               thrust::device_vector<float>& d_Y) -> Result
{
    const int grid = grid_for(v, m);

    auto launch = [&] {
        v.kernel<<<grid, kBlockSize>>>(
            1.0f,
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_x.data()),
            0.0f,   // beta = 0, so repeated launches stay idempotent
            thrust::raw_pointer_cast(d_Y.data()),
            m, n);
    };

    for (int i = 0; i < kWarmup; ++i) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // hipEvent rather than std::chrono: it timestamps on the device, so host
    // launch overhead stays out of the measurement.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < kIters; ++i) launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    HIP_CHECK(hipDeviceSynchronize());

    const double useful = static_cast<double>(m) * n + n + 2.0 * m;
    const double bytes  = useful * sizeof(float);
    const double secs   = (total_ms / 1e3) / kIters;

    return {secs * 1e3, bytes / secs, bytes / secs / kPeakBps};
}

} // namespace

auto main(int argc, char** argv) -> int
{
    const int m = 1 << 20;
    const int n = 1 << 10;

    std::vector<Variant> selected;
    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            const Variant* v = find_variant(argv[i]);
            if (!v)
            {
                std::cerr << "unknown variant: " << argv[i] << "\navailable:";
                for (const auto& c : variants()) std::cerr << ' ' << c.name;
                std::cerr << '\n';
                return EXIT_FAILURE;
            }
            selected.push_back(*v);
        }
    }
    else
    {
        selected = variants();
    }

    try
    {
        std::vector<float> A(static_cast<size_t>(m) * n);
        std::vector<float> x(n);
        std::vector<float> Y(m);

        std::mt19937 gen{42};
        std::uniform_real_distribution<float> dis{-1.0f, 1.0f};
        std::generate(A.begin(), A.end(), [&] { return dis(gen); });
        std::generate(x.begin(), x.end(), [&] { return dis(gen); });
        std::generate(Y.begin(), Y.end(), [&] { return dis(gen); });

        thrust::device_vector<float> d_A = A;
        thrust::device_vector<float> d_x = x;
        thrust::device_vector<float> d_Y = Y;

        const double gb = (static_cast<double>(m) * n + n + 2.0 * m) * sizeof(float) / 1e9;
        std::cout << "m x n = " << m << " x " << n << "  (" << std::fixed
                  << std::setprecision(3) << gb << " GB per launch)\n"
                  << kIters << " iterations, +" << kWarmup << " warm-up, hipEvent timing\n\n";

        std::cout << std::left << std::setw(10) << "variant"
                  << std::right << std::setw(8) << "thr/row"
                  << std::setw(12) << "time (ms)"
                  << std::setw(12) << "GB/s"
                  << std::setw(10) << "% of peak" << '\n';

        for (const auto& v : selected)
        {
            const Result r = benchmark(v, m, n, d_A, d_x, d_Y);
            std::cout << std::left << std::setw(10) << v.name
                      << std::right << std::setw(8) << v.threads_per_row
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.ms
                      << std::setw(12) << std::setprecision(1) << r.bw / 1e9
                      << std::setw(9) << std::setprecision(1) << r.efficiency * 100 << '%'
                      << '\n';
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Erreur Thrust/C++ : " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
