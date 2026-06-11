# `P8_SIGMOID.cu` Results

Updated from regenerated local logs:

- CPU-backed correctness run: [p8_with_cpu.txt](p8_with_cpu.txt)
- Benchmark run: [p8_skip_cpu.txt](p8_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep: not generated yet

## Kernel Variants

- `basic`: scalar grid-stride Sigmoid kernel
- `float4`: vectorized `float4` Sigmoid kernel with scalar tail handling

## Correctness Summary

From [p8_with_cpu.txt](p8_with_cpu.txt):

- `4 PASS/PASS` rows, `12 REF/PASS` rows
- `0 FAIL` rows
- Both kernels pass all small exact tests.
- Both kernels pass generated medium and large CPU-reference cases.

From [p8_skip_cpu.txt](p8_skip_cpu.txt):

- `4 SKIP/PASS` rows, `10 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara rows are benchmark-only.

## Performance Summary

From [p8_skip_cpu.txt](p8_skip_cpu.txt):

- Across published Tensara benchmark rows, `float4` wins `5` of `5`.
- At default launch, `float4` is about `6%` faster than `basic` on the
  published shapes.
- No launch sweep has been generated yet, so `(256, 64)` is the only measured
  launch shape for P8.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`
  - default best: `float4 = 0.757 ms`
- `tensara_2`: `6144 x 4096`
  - default best: `float4 = 1.132 ms`
- `tensara_3`: `4096 x 7168`
  - default best: `float4 = 1.317 ms`
- `tensara_4`: `4096 x 8192`
  - default best: `float4 = 1.510 ms`
- `tensara_5`: `8192 x 8192`
  - default best: `float4 = 3.019 ms`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.806`, `float4 0.757`
- `tensara_2`: `basic 1.210`, `float4 1.132`
- `tensara_3`: `basic 1.405`, `float4 1.317`
- `tensara_4`: `basic 1.604`, `float4 1.510`
- `tensara_5`: `basic 3.230`, `float4 3.019`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
- P8 has no saved launch sweep yet.
