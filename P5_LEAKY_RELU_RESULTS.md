# `P5_LEAKY_RELU.cu` Results

Updated from regenerated local logs:

- CPU-backed correctness run: [p5_with_cpu.txt](p5_with_cpu.txt)
- Benchmark run: [p5_skip_cpu.txt](p5_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Kernel Variants

- `basic`: scalar grid-stride Leaky ReLU kernel
- `float4`: vectorized `float4` kernel with scalar tail handling

## Correctness Summary

From [p5_with_cpu.txt](p5_with_cpu.txt):

- `6 PASS/PASS` rows, `94 REF/PASS` rows
- `0 FAIL` rows
- Both kernels pass all small exact tests.
- Both kernels pass medium and larger CPU-reference cases.
- Both kernels pass CPU-backed launch sweeps.

From [p5_skip_cpu.txt](p5_skip_cpu.txt):

- `6 SKIP/PASS` rows, `484 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara, shape, tail, and scale rows are benchmark-only.

## Performance Summary

From [p5_skip_cpu.txt](p5_skip_cpu.txt):

- Across all comparable skip-CPU configurations, `float4` wins `240`
  of `245`; `basic` wins `3`; `2` are ties.
- Across default-launch non-scale rows, `float4` wins `20` of `25`;
  `basic` wins `3`; `2` are ties.
- `float4` wins every published Tensara row.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`, `alpha=0.01`
  - default best: `float4 = 0.756 ms`
  - scale best: `float4 = 0.754 ms` at `(64, 128)`
- `tensara_2`: `4096 x 4096`, `alpha=0.05`
  - default best: `float4 = 0.753 ms`
  - scale best: `float4 = 0.752 ms` at `(128, 128)`
- `tensara_3`: `4096 x 4096`, `alpha=0.10`
  - default best: `float4 = 0.753 ms`
  - scale best: `float4 = 0.751 ms` at `(128, 128)`
- `tensara_4`: `4096 x 4096`, `alpha=0.20`
  - default best: `float4 = 0.753 ms`
  - scale best: `float4 = 0.751 ms` at `(128, 128)`
- `tensara_5`: `6144 x 4096`, `alpha=0.01`
  - default best: `float4 = 1.125 ms`
  - scale best: `float4 = 1.119 ms` at `(512, 128)`
- `tensara_6`: `6144 x 4096`, `alpha=0.05`
  - default best: `float4 = 1.125 ms`
  - scale best: `float4 = 1.120 ms` at `(512, 128)`
- `tensara_7`: `6144 x 4096`, `alpha=0.10`
  - default best: `float4 = 1.126 ms`
  - scale best: `float4 = 1.119 ms` at `(512, 128)`
- `tensara_8`: `6144 x 4096`, `alpha=0.20`
  - default best: `float4 = 1.129 ms`
  - scale best: `float4 = 1.121 ms` at `(512, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.794`, `float4 0.756`
- `tensara_2`: `basic 0.793`, `float4 0.753`
- `tensara_3`: `basic 0.796`, `float4 0.753`
- `tensara_4`: `basic 0.794`, `float4 0.753`
- `tensara_5`: `basic 1.195`, `float4 1.125`
- `tensara_6`: `basic 1.190`, `float4 1.125`
- `tensara_7`: `basic 1.188`, `float4 1.126`
- `tensara_8`: `basic 1.193`, `float4 1.129`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
