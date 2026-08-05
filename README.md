# SGEMV — Matrix-Vector Multiplication on RDNA3

My master's thesis benchmarked successive optimization levels
of a SYCL SGEMV against an optimized CUDA implementation. This project
revisits the problem on AMD hardware, with bandwidth utilization as
the primary metric.


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

## Results

## Kernel variants
- naive
## Options to run
```bash
make build
make run
make clean
```
