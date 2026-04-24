# `P3_RELU.cu` Results

Detailed local benchmark notes and raw output for the baseline ReLU kernel.

## Kernel Variant

- `basic`: baseline grid-stride ReLU kernel

## Local Benchmark Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch for the main benchmark rows: `block_x=256`, `grid_x=64`
- Scaling sweep launches:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Important Caveat

This result file appears to come from a `--skip-cpu` run:

- All rows show `cpu=SKIP`
- Only the `small_*` rows show `gpu=PASS`
- The `medium`, `large`, `shape`, and `scale` rows were timed, but they were
  not checked against a CPU reference in this capture

That means this file is useful for performance interpretation and launch-shape
comparison, but it is not yet a full correctness report for the larger cases.

## Summary

- The baseline ReLU kernel passes the current hand-written small tests.
- Kernel time scales close to linearly with total element count, which is what
  you would expect for a simple memory-bound elementwise kernel.
- The square, wide, and tall scaling sweeps all prefer high overall launch
  concurrency; weak launches like `64 x 8` are clearly underfilled.
- The best results in this run cluster around `grid_x=128`, with `block_x=128`
  or `256` usually at or near the minimum.
- The default launch `256 x 64` is already close to optimal for the square case,
  but it is not the absolute best point from the scaling sweep.

## Representative Default-Launch Rows

Using the default `block_x=256`, `grid_x=64` rows:

- `medium_2 (256 x 256)`: `kernel_ms=0.006`
- `medium_4 (1024 x 1024)`: `kernel_ms=0.054`
- `large_1 (2048 x 1024)`: `kernel_ms=0.104`
- `large_2 (4096 x 1024)`: `kernel_ms=0.203`
- `shape_1 (64 x 16384)`: `kernel_ms=0.055`
- `shape_2 (16384 x 64)`: `kernel_ms=0.055`

Two useful interpretations:

- Wide and tall matrices with the same total number of elements perform almost
  identically, which is a good sign for a flat elementwise traversal.
- For larger cases, `total_ms` is much bigger than `kernel_ms`, so host-side
  allocation/copy overhead still matters a lot in this harness.

## Best Scaling Results

Using `kernel_ms` as the main metric:

- `scale_sq / basic`: best `0.054 ms` at `(block_x=128, grid_x=128)`
- `scale_wide / basic`: best `0.202 ms` at `(block_x=128, grid_x=128)`
- `scale_tall / basic`: best `0.202 ms` at `(block_x=128, grid_x=128)`

These best points are not unique in every case, but they give a clean default
direction: for this GPU and this kernel, `128 x 128` is a strong launch shape.

## Raw Results

The interpreted sections above are summaries. The sections below keep the raw
console-style data in a form that is easy to scan and cross-check.

### Raw Scaling Heatmaps

These are compact heatmaps for `kernel_ms` with lower-is-better semantics.

```text
scale_sq / basic  best=(128,128) 0.054 ms
block\grid8         16        32        64        128
64        0.781     0.386     0.201     0.113     0.068
128       0.386     0.202     0.110     0.070     0.054
256       0.202     0.110     0.069     0.054     0.060
512       0.111     0.070     0.054     0.060     0.056

scale_wide / basic  best=(128,128) 0.202 ms
block\grid8         16        32        64        128
64        3.026     1.527     0.788     0.423     0.257
128       1.531     0.788     0.424     0.254     0.202
256       0.793     0.425     0.256     0.203     0.227
512       0.426     0.256     0.202     0.230     0.209

scale_tall / basic  best=(128,128) 0.202 ms
block\grid8         16        32        64        128
64        3.031     1.532     0.790     0.426     0.257
128       1.529     0.787     0.426     0.254     0.202
256       0.793     0.425     0.257     0.203     0.229
512       0.425     0.254     0.202     0.249     0.206
```

## What Seems To Matter

- Increasing `grid_x` helps a lot at small and medium block sizes because the
  GPU gets enough total work to hide latency.
- Once the launch is strong enough, larger `block_x` values are not uniformly
  better. The best region is broad rather than a single razor-thin optimum.
- `block_x=64, grid_x=8` is consistently poor and should be avoided.
- `block_x=128` is a strong baseline choice because it matches the best point in
  all three scaling cases without requiring the largest block size.

## Recommended Next Runs

- Run the default harness path without `--skip-cpu` and save that output as a
  separate correctness-oriented result file.
- Keep this file as the benchmark-oriented `--skip-cpu` run, since it is useful
  for launch tuning.
- After adding new kernel variants, extend this format so each variant gets:
  - one short summary bullet
  - one best-scaling line
  - one shared raw-output block

### Raw Benchmark Output

```text
group   name          kernel        rows      cols      block_x grid_x  cpu   gpu   total_ms    kernel_ms
------------------------------------------------------------------------------------------------------------
small   small_1       basic         2         4         256     64      SKIP  PASS  2.789       1.750
small   small_2       basic         3         3         256     64      SKIP  PASS  0.339       0.004
small   small_3       basic         1         8         256     64      SKIP  PASS  0.381       0.005
medium  medium_1      basic         64        64        256     64      SKIP  SKIP  0.910       0.008
medium  medium_2      basic         256       256       256     64      SKIP  SKIP  0.449       0.006
medium  medium_3      basic         512       1024      256     64      SKIP  SKIP  1.070       0.030
medium  medium_4      basic         1024      1024      256     64      SKIP  SKIP  1.789       0.054
large   large_1       basic         2048      1024      256     64      SKIP  SKIP  3.711       0.104
large   large_2       basic         4096      1024      256     64      SKIP  SKIP  6.272       0.203
shape   shape_1       basic         64        16384     256     64      SKIP  SKIP  1.944       0.055
shape   shape_2       basic         16384     64        256     64      SKIP  SKIP  1.844       0.055
shape   shape_3       basic         512       8192      256     64      SKIP  SKIP  6.098       0.202
shape   shape_4       basic         8192      512       256     64      SKIP  SKIP  5.875       0.203
scale   scale_sq      basic         1024      1024      64      8       SKIP  SKIP  3.034       0.781
scale   scale_sq      basic         1024      1024      64      16      SKIP  SKIP  1.998       0.386
scale   scale_sq      basic         1024      1024      64      32      SKIP  SKIP  2.105       0.201
scale   scale_sq      basic         1024      1024      64      64      SKIP  SKIP  2.160       0.113
scale   scale_sq      basic         1024      1024      64      128     SKIP  SKIP  1.590       0.068
scale   scale_sq      basic         1024      1024      128     8       SKIP  SKIP  1.946       0.386
scale   scale_sq      basic         1024      1024      128     16      SKIP  SKIP  1.702       0.202
scale   scale_sq      basic         1024      1024      128     32      SKIP  SKIP  1.610       0.110
scale   scale_sq      basic         1024      1024      128     64      SKIP  SKIP  1.581       0.070
scale   scale_sq      basic         1024      1024      128     128     SKIP  SKIP  1.671       0.054
scale   scale_sq      basic         1024      1024      256     8       SKIP  SKIP  1.713       0.202
scale   scale_sq      basic         1024      1024      256     16      SKIP  SKIP  1.592       0.110
scale   scale_sq      basic         1024      1024      256     32      SKIP  SKIP  1.536       0.069
scale   scale_sq      basic         1024      1024      256     64      SKIP  SKIP  1.529       0.054
scale   scale_sq      basic         1024      1024      256     128     SKIP  SKIP  1.523       0.060
scale   scale_sq      basic         1024      1024      512     8       SKIP  SKIP  1.926       0.111
scale   scale_sq      basic         1024      1024      512     16      SKIP  SKIP  1.705       0.070
scale   scale_sq      basic         1024      1024      512     32      SKIP  SKIP  1.569       0.054
scale   scale_sq      basic         1024      1024      512     64      SKIP  SKIP  1.513       0.060
scale   scale_sq      basic         1024      1024      512     128     SKIP  SKIP  1.568       0.056
scale   scale_wide    basic         256       16384     64      8       SKIP  SKIP  8.996       3.026
scale   scale_wide    basic         256       16384     64      16      SKIP  SKIP  7.352       1.527
scale   scale_wide    basic         256       16384     64      32      SKIP  SKIP  6.218       0.788
scale   scale_wide    basic         256       16384     64      64      SKIP  SKIP  7.580       0.423
scale   scale_wide    basic         256       16384     64      128     SKIP  SKIP  5.720       0.257
scale   scale_wide    basic         256       16384     128     8       SKIP  SKIP  6.842       1.531
scale   scale_wide    basic         256       16384     128     16      SKIP  SKIP  6.315       0.788
scale   scale_wide    basic         256       16384     128     32      SKIP  SKIP  5.600       0.424
scale   scale_wide    basic         256       16384     128     64      SKIP  SKIP  5.565       0.254
scale   scale_wide    basic         256       16384     128     128     SKIP  SKIP  5.651       0.202
scale   scale_wide    basic         256       16384     256     8       SKIP  SKIP  6.689       0.793
scale   scale_wide    basic         256       16384     256     16      SKIP  SKIP  7.241       0.425
scale   scale_wide    basic         256       16384     256     32      SKIP  SKIP  5.544       0.256
scale   scale_wide    basic         256       16384     256     64      SKIP  SKIP  6.291       0.203
scale   scale_wide    basic         256       16384     256     128     SKIP  SKIP  6.336       0.227
scale   scale_wide    basic         256       16384     512     8       SKIP  SKIP  5.824       0.426
scale   scale_wide    basic         256       16384     512     16      SKIP  SKIP  5.738       0.256
scale   scale_wide    basic         256       16384     512     32      SKIP  SKIP  5.471       0.202
scale   scale_wide    basic         256       16384     512     64      SKIP  SKIP  5.427       0.230
scale   scale_wide    basic         256       16384     512     128     SKIP  SKIP  5.558       0.209
scale   scale_tall    basic         16384     256       64      8       SKIP  SKIP  9.974       3.031
scale   scale_tall    basic         16384     256       64      16      SKIP  SKIP  7.618       1.532
scale   scale_tall    basic         16384     256       64      32      SKIP  SKIP  6.462       0.790
scale   scale_tall    basic         16384     256       64      64      SKIP  SKIP  6.067       0.426
scale   scale_tall    basic         16384     256       64      128     SKIP  SKIP  6.074       0.257
scale   scale_tall    basic         16384     256       128     8       SKIP  SKIP  6.923       1.529
scale   scale_tall    basic         16384     256       128     16      SKIP  SKIP  6.027       0.787
scale   scale_tall    basic         16384     256       128     32      SKIP  SKIP  5.564       0.426
scale   scale_tall    basic         16384     256       128     64      SKIP  SKIP  6.034       0.254
scale   scale_tall    basic         16384     256       128     128     SKIP  SKIP  5.524       0.202
scale   scale_tall    basic         16384     256       256     8       SKIP  SKIP  6.202       0.793
scale   scale_tall    basic         16384     256       256     16      SKIP  SKIP  5.957       0.425
scale   scale_tall    basic         16384     256       256     32      SKIP  SKIP  5.348       0.257
scale   scale_tall    basic         16384     256       256     64      SKIP  SKIP  6.165       0.203
scale   scale_tall    basic         16384     256       256     128     SKIP  SKIP  5.589       0.229
scale   scale_tall    basic         16384     256       512     8       SKIP  SKIP  5.566       0.425
scale   scale_tall    basic         16384     256       512     16      SKIP  SKIP  5.303       0.254
scale   scale_tall    basic         16384     256       512     32      SKIP  SKIP  5.972       0.202
scale   scale_tall    basic         16384     256       512     64      SKIP  SKIP  5.374       0.249
scale   scale_tall    basic         16384     256       512     128     SKIP  SKIP  5.844       0.206
```
