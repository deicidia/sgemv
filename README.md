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
| Practical peak (BabelStream 5.0, Dot, float, 403 MB) | **528** |

Efficiency figures below will use the **practical** peak as denominator.
Dot is the closest structural analogue to SGEMV: read-dominated,
with a reduction and a negligible write.

    useful_bytes = (M·N + N + 2M) · sizeof(float)
    BW_eff       = useful_bytes / elapsed
    efficiency   = BW_eff / 528e9

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
| `sgemv_wave32` | — | — | — |

## Results

| Variant | Time | BW_eff | % of 528 |
|---|---|---|---|
| `sgemv_naive` | 210.19 ms | 20 GB/s | **3.9%** |
| `sgemv_block` | 8.79 ms | 490 GB/s | **92.7 %** | 
| `sgemv_wave32` | — | — | — |

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



