# SGEMV — Matrix-Vector Multiplication on RDNA3

My master's thesis benchmarked successive optimization levels
of a SYCL SGEMV against an optimized CUDA implementation. This project
revisits the problem on AMD hardware, with bandwidth utilization as
the primary metric.

My previous implementations stopped at a suboptimal `sgemv_block`, and I will focus here on `sgemv_wave32` and on using the `float4` type for wider load/store instructions.

## Metric

| | GB/s |
|---|---|
| Theoretical peak (GDDR6) | **624** |
| Practical peak (BabelStream 5.0, Copy, float, 403 MB) | **613**|

The denominator started as BabelStream's Dot (521 GB/s), the closest
structural analogue: read-dominated, with a reduction and a negligible write. However, since `sgevm_wave32` reached 111% of this baseline, I switched to Copy (613 GB/s). This is the highest bandwidth BabelStream achieves on this card, providing a proper upper bound for a read-dominated kernel.

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

## Results

| Variant | Time | BW_eff | % of 613 |
|---|---|---|---|
| `sgemv_naive` | 189.84 ms | 22.7 GB/s | **3.7 %** |
| `sgemv_block` | 	8.75 ms | 492.0 GB/s | **80.3 %** | 
| `sgemv_wave32` | 7.29 ms | 589.9 GB/s | **96.2 %** |

### Performance Analysis:

The `sgemv_wave32` variant achieves **589.9 GB/s** (96.2% of peak bandwidth) by maximizing memory-level parallelism.

*   **Memory-Level Parallelism (MLP):** The `#pragma unroll 4` directive is the primary driver of this performance jump. It allows the GPU to issue multiple memory loads concurrently before waiting on the data, effectively hiding memory latency and avoiding pipeline stalls.
*   **The Vectorization Limit:** The compiler does not emit 128-bit vectorized loads (`global_load_b128`). Because the loop increments by `warpSize`, a single thread's successive memory reads are separated by 128 bytes. Without contiguous memory access per thread, the compiler cannot merge these loads.
*   **Future Optimization:** To unlock vectorized 128-bit loads, the memory access pattern needs to be restructured using `float4`. This requires each thread to read four adjacent floats simultaneously rather than striding across the warp.

## Reproducing

Variants are declared once, in `include/variants.hpp`:

```cpp
{"naive",  sgemv_naive,    1},   // name, kernel, threads per row
{"block",  sgemv_block,  256},
{"wave32", sgemv_wave32,  32},
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



