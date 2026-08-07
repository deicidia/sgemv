# SGEMV — Matrix-Vector Multiplication on RDNA3

My master's thesis benchmarked successive optimization levels
of a SYCL SGEMV against an optimized CUDA implementation. This project
revisits the problem on AMD hardware, with bandwidth utilization as
the primary metric.

My previous implementations stopped at a suboptimal `sgemv_block`, this repository focuses on using one wave per row, using the `float4` type for wider load/store instructions and `pragma unroll 4` for Memory-Level Parallelism.

## Metric

| | GB/s |
|---|---|
| Theoretical peak (GDDR6) | **624** |
| Practical peak (BabelStream 5.0, Copy, float, 134 MB/array) | **613**|

The denominator started as BabelStream's Dot (521 GB/s), the closest
structural analogue: read-dominated, with a reduction and a negligible write. However, since `sgemv_wave32` reached 111% of this baseline, I switched to Copy (613 GB/s). This is the highest bandwidth BabelStream achieves on this card, providing a proper upper bound for a read-dominated kernel.

    useful_bytes = (M·N + N + 2M) · sizeof(float)
    BW_eff       = useful_bytes / elapsed
    efficiency   = BW_eff / 613e9

## Environment

| | |
|---|---|
| **GPU** | AMD Radeon RX 7800 XT — Navi 32, `0x747e` |
| **Architecture** | gfx1101 (RDNA3), 60 CU, wave32 |
| **Engine clock** | 2169 MHz max |
| **VRAM** | 16 368 MB GDDR6 (Samsung), 256-bit |
| **Memory clock** | 1218 MHz (DPM state 3, max) |
| **Peak bandwidth** | 1218 MHz × 16 × 256 bit / 8 = **624 GB/s** |
| **Cacheline** | 128 B |
| **PCIe** | Gen4 ×16 (16 GT/s) |
| **Board power** | 220 W |
| **Host** | Ryzen 5 5600 (6C/12T), Ubuntu 24.04.4, kernel 7.0.0-28 |
| **Driver** | amdgpu 6.19.14.31400000 |
| **ROCm** | amdrocm 7.14.0~pre3-29052710811 (pre-release) |
| **HIP / compiler** | HIP 7.14.60850, AMD clang 23.0.0git (`46fcb339`) |
| **Profiler** | rocprofv3 1.3.2 |

### Cache hierarchy

| Level | Size | Shared by | Role |
|---|---|---|---|
| **L0 vector** | 32 KB | 1 CU | per-CU data |
| **L0 scalar** | 16 KB | 2 CU | per-WGP |
| **L0 instruction** | 32 KB | 2 CU | per-WGP |
| **L1** | 256 KB | 10 CU | per shader array |
| **L2** | 4 MB | 60 CU | device-wide |
| **Infinity Cache (MALL)** | 64 MB | 60 CU | in front of GDDR6 |

## Kernel variants

| Variant | Geometry | Reduction | LDS/block | VGPR |
|---|---|---|---|---|
| `sgemv_naive` | 1 thread per row | sequential, in-register | 0 | 5 |
| `sgemv_block` | 1 block (256 thr) per row | LDS tree, then wave shuffles | 1 KB | 8 |
| `sgemv_wave32` | 1 wave (32 thr) per row | wave shuffles | 0 | 17 |
| `sgemv_wave32_float4` | 1 wave (32 thr) per row | wave shuffles | 0 | 40 |

## Results

| Variant | Time | BW_eff | % of 613 |
|---|---|---|---|
| `sgemv_naive` | 189.84 ms | 22.7 GB/s | **3.7 %** |
| `sgemv_block` | 8.75 ms | 492.0 GB/s | **80.3 %** | 
| `sgemv_wave32` | 7.29 ms | 589.9 GB/s | **96.2 %** |
| `sgemv_wave32_float4` | 7.11 ms | 605.0 GB/s | **98.7 %** |

Benchmarked using $m = 2^{20}$ and $n = 2^{10}$. 4.303 GB of incompressible traffic per launch, well clear
of the 64 MB Infinity Cache. Mean over 50 launches after 5 warm-up launches,
timed with `hipEvent` so that host launch overhead and the D2H copy stay out
of the measurement. A single cold launch swings from 78 % to 94 %, which is what the warm-up and the loop buy.

### Performance Analysis:

The `sgemv_wave32_float4` variant achieves **605.0 GB/s** (98.7% of peak bandwidth) by maximizing memory-level parallelism through concurrent memory loads. Utilizing the `float4` data type enables the compiler to generate `global_load_b128` instructions without requiring inline assembly. Furthermore, assigning a single wavefront per row minimizes the need for block-wide synchronization (__syncthreads()) and leverages the `__shfl_xor` intrinsic to completely avoid inter-wave communication during the kernel's reduction phase.

## Reproducing

Variants are declared once, in `include/variants.hpp`:

```cpp
{"naive",  sgemv_naive,    1},   // name, kernel, threads per row
{"block",  sgemv_block,  256},
{"wave32", sgemv_wave32,  32},
{"wave32_float4", sgemv_wave32_float4,  32},
```

Both the benchmark and the test binary read that table, and both derive the
launch geometry from `threads_per_row` rather than spelling it out. Adding a
variant means one line here and one kernel. The harness does not change, and
the grid cannot drift apart between the two binaries.

```bash
make bench                   # every variant
make bench VARIANT=naive     # one, or several: VARIANT="naive block"
make test                    # correctness: every variant x 5 shapes
make resources               # LDS, VGPR, spills, occupancy
make isa                     # demangled GCN assembly
make clean
```



