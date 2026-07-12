# `P5_LEAKY_RELU.cu` Results

## At a Glance

- Recommendation: `float4`.
- Correctness: all 100 current CPU-backed rows pass.
- Performance: wins 63 of 73 comparable current benchmark table rows.
- Status: current raw logs; benchmark rows remain unverified.

Current skip-CPU evidence has 6 exact checked rows and 484 benchmark-only
rows. `float4` wins all eight published Tensara rows.

Benchmark-only rows are timing evidence, not correctness evidence. See the
[repository index](RESULTS_INDEX.md) for cross-problem status.

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

- Across comparable table rows, `float4` wins `63` of `73`; `basic` wins `2`;
  `8` are ties.
- Across published Tensara rows, `float4` wins `8` of `8`.
- `float4` wins every published Tensara row.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`, `alpha=0.01`
  - default best: `float4 = 0.752 ms`
  - scale best: `float4 = 0.751 ms` at `(512, 128)`
- `tensara_2`: `4096 x 4096`, `alpha=0.05`
  - default best: `float4 = 0.752 ms`
  - scale best: `float4 = 0.753 ms` at `(64, 128)`
- `tensara_3`: `4096 x 4096`, `alpha=0.10`
  - default best: `float4 = 0.755 ms`
  - scale best: `float4 = 0.750 ms` at `(512, 128)`
- `tensara_4`: `4096 x 4096`, `alpha=0.20`
  - default best: `float4 = 0.754 ms`
  - scale best: `float4 = 0.751 ms` at `(128, 128)`
- `tensara_5`: `6144 x 4096`, `alpha=0.01`
  - default best: `float4 = 1.126 ms`
  - scale best: `float4 = 1.120 ms` at `(512, 128)`
- `tensara_6`: `6144 x 4096`, `alpha=0.05`
  - default best: `float4 = 1.128 ms`
  - scale best: `float4 = 1.121 ms` at `(512, 128)`
- `tensara_7`: `6144 x 4096`, `alpha=0.10`
  - default best: `float4 = 1.125 ms`
  - scale best: `float4 = 1.119 ms` at `(512, 128)`
- `tensara_8`: `6144 x 4096`, `alpha=0.20`
  - default best: `float4 = 1.135 ms`
  - scale best: `float4 = 1.119 ms` at `(512, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.797`, `float4 0.752`
- `tensara_2`: `basic 0.792`, `float4 0.752`
- `tensara_3`: `basic 0.792`, `float4 0.755`
- `tensara_4`: `basic 0.792`, `float4 0.754`
- `tensara_5`: `basic 1.182`, `float4 1.126`
- `tensara_6`: `basic 1.187`, `float4 1.128`
- `tensara_7`: `basic 1.186`, `float4 1.125`
- `tensara_8`: `basic 1.184`, `float4 1.135`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
