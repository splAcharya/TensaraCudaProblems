# `P3_RELU.cu` Results

Updated summary based on the current two-variant ReLU harness:

- CPU-backed correctness run: [p3_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_with_cpu.txt)
- Heavier benchmark run: [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt)

## Kernel Variants

- `basic`: scalar grid-stride ReLU kernel
- `float4`: vectorized `float4` kernel with scalar tail handling

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`
- Default runs now verify:
  - small exact tests
  - medium generated tests
  - selected larger odd-shape cases
- `--skip-cpu` adds:
  - Tensara-size rows
  - shape variants
  - dedicated odd tail cases
  - scaling sweeps

## Correctness Summary

From [p3_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_with_cpu.txt):

- Status legend:
  - `cpu=PASS`: CPU output matched a hard-coded expected answer.
  - `cpu=REF`: CPU output was generated and used as the reference for GPU verification.
  - `cpu=SKIP`: CPU reference generation was skipped.
  - `gpu=PASS`: GPU output matched either the hard-coded expected answer or the CPU reference.
  - `gpu=SKIP`: GPU verification was skipped, usually because the run used `--skip-cpu`.
- `basic` passes all current small tests.
- `float4` also passes all current small tests.
- Both kernels pass the medium CPU-reference cases, including odd matrix sizes:
  - `255 x 257`
  - `513 x 1025`
- Both kernels also pass the selected larger odd-shape verification cases:
  - `1023 x 2049`
  - `1537 x 2049`

That is the main result of the recent harness work: the `float4` path is now being checked on nontrivial odd shapes instead of only on tiny toy cases.

## Performance Summary

From [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt):

- On most default-launch rows, `float4` is modestly faster than `basic`.
- The new `tail_*` rows also favor `float4`, which is a useful sign that the scalar remainder path is not obviously hurting the kernel.
- The scaling sweeps consistently show a small but repeatable `float4` advantage.

Representative default-launch rows:

- `tensara_1 (4096 x 4096)`:
  - `basic 0.794 ms`
  - `float4 0.754 ms`
- `tensara_2 (6144 x 4096)`:
  - `basic 1.192 ms`
  - `float4 1.124 ms`
- `tensara_3 (4096 x 7168)`:
  - `basic 1.379 ms`
  - `float4 1.310 ms`
- `tensara_4 (4096 x 8192)`:
  - `basic 1.572 ms`
  - `float4 1.498 ms`
- `tensara_5 (8192 x 8192)`:
  - `basic 3.201 ms`
  - `float4 2.982 ms`
- `tail_1 (2049 x 2049)`:
  - `basic 0.200 ms`
  - `float4 0.193 ms`
- `tail_2 (3073 x 4097)`:
  - `basic 0.589 ms`
  - `float4 0.565 ms`
- `tail_3 (4097 x 8193)`:
  - `basic 1.562 ms`
  - `float4 1.499 ms`

## Rerun Note

The retained skip-CPU run is the clean rerun:

- `tensara_5 (8192 x 8192)` reports `float4 = 2.982 ms`
- `shape_4 (8192 x 8192)` reports `float4 = 2.990 ms`

Earlier reruns showed intermittent `shape_4 / float4` outliers near `29 ms`, while `tensara_5 / float4` stayed near `3 ms` in the clean reruns. Because `tensara_5` and `shape_4` use the same `8192 x 8192` shape and default launch, the outlier appears to be benchmark noise rather than a separate kernel behavior.

The safer reading is:

- `float4` looks correct
- `float4` is slightly faster than `basic` on the verified medium, large, tail, and Tensara-sized rows
- the representative `8192 x 8192` `float4` time is about `3.0 ms`, not `29 ms`

## Best Scaling Results

From the heatmaps in [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt):

- `scale_sq / basic`: best `0.782 ms` at `(256, 64)`
- `scale_sq / float4`: best `0.752 ms` at `(64, 128)`
- `scale_rect_1 / basic`: best `1.169 ms` at `(256, 64)`
- `scale_rect_1 / float4`: best `1.121 ms` at `(256, 64)`
- `scale_rect_2 / basic`: best `1.552 ms` at `(256, 64)`
- `scale_rect_2 / float4`: best `1.497 ms` at `(512, 16)`

The broad pattern is:

- weak launches such as low-grid, low-block combinations are clearly underfilled
- once the launch is strong enough, the `float4` kernel tends to shave a small amount off the scalar baseline
- the best region is fairly broad rather than razor-thin

## Notes

- The markdown summary is intentionally shorter than the raw console dumps.
- For complete row-by-row data, use:
  - [p3_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_with_cpu.txt)
  - [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt)
