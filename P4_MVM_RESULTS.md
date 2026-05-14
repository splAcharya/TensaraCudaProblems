# `P4_MVM.cu` Results

Updated summary based on six-variant CPU-backed and skip-CPU logs using
in-harness median timing.

- CPU-backed correctness run: [p4_with_cpu.txt](p4_with_cpu.txt)
- Heavier benchmark run: [p4_skip_cpu.txt](p4_skip_cpu.txt)
- Follow-up ideas: [P4_MVM_OPTIMIZATION_NOTES.md](P4_MVM_OPTIMIZATION_NOTES.md)

## Kernel Variants

- `basic`: direct global-memory implementation
- `constant_b`: direct implementation with `B` in constant memory
- `shared_ab`: shared-memory tiles for `A` and `B`
- `warp`: register accumulation with warp-level reduction
- `warp_const_b`: same warp-level mapping as `warp`, but with `B` preloaded
  into constant memory
- `warp_per_row`: one warp computes one output row; one block computes
  multiple output rows

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: each row reports median `kernel_ms` from `5` timed samples after
  `1` warmup launch.
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`
- Default runs verify:
  - small exact tests
  - generated medium cases
  - selected larger cases
  - a CPU-backed launch sweep
- `--skip-cpu` adds:
  - Tensara-size rows
  - tall, wide, odd, and rectangular shapes
  - larger launch-configuration sweeps

## Correctness Summary

From [p4_with_cpu.txt](p4_with_cpu.txt):

- All six kernels pass the small exact tests.
- All six kernels pass the CPU-reference medium and large cases.
- The CPU-backed launch sweep passes for every tested variant.
- The refreshed CPU-backed log has `24 PASS/PASS`, `162 REF/PASS`,
  `0 FAIL`, and `0 SKIP` result rows.

From [p4_skip_cpu.txt](p4_skip_cpu.txt):

- The skip-CPU benchmark log has `24 SKIP/PASS`, `456 SKIP/SKIP`,
  and `0 FAIL` result rows.
- The small exact rows are still checked even in skip-CPU mode.
- Larger Tensara and shape rows are benchmark-only in this log.

## Performance Summary

From [p4_with_cpu.txt](p4_with_cpu.txt), which is correctness-oriented:

- `warp_per_row` wins 24 of 31 comparable CPU-backed configurations.
- `constant_b` wins 1 configuration.
- `warp` wins 1 configuration.
- `basic`, `shared_ab`, and `warp_const_b` do not win any unique
  configuration.
- There are 5 ties.

From [p4_skip_cpu.txt](p4_skip_cpu.txt):

- `warp_per_row` wins 54 of 80 comparable benchmark configurations.
- `constant_b` wins 9 configurations.
- `warp` wins 7 configurations.
- `basic` wins 1 configuration.
- `warp_const_b` wins 1 configuration.
- `shared_ab` does not win any unique configuration.
- There are 8 ties.

The broad result changed after adding `warp_per_row`: it is now the best
overall benchmark variant in the launch sweeps. On fixed default-launch
Tensara rows, `warp_per_row` now wins all five rows under in-harness median
timing, though the margin over `warp` is small. `constant_b` helps relative
to `basic` on larger fixed-launch rows. `warp_const_b` remains a useful
negative-result variant, and `shared_ab` is correct but usually pays too much
synchronization and shared-memory overhead.

## Tensara-Size Rows

From [p4_skip_cpu.txt](p4_skip_cpu.txt), default launch, `kernel_ms`:

```text
name       shape       basic  const_b  shared  warp   warp_c  warp_row
tensara_1 4096x4096   1.482  1.077    2.431   0.366  1.889   0.362
tensara_2 6144x4096   1.530  1.507    3.644   0.545  2.777   0.542
tensara_3 7168x4096   1.913  1.727    4.251   0.635  3.254   0.631
tensara_4 8192x4096   2.051  1.965    4.858   0.725  3.742   0.722
tensara_5 9216x4096   2.292  2.234    5.463   0.815  4.196   0.812
```

`warp_per_row` wins all five default-launch Tensara rows in the median-timed
log. The two warp-level kernels are still close on the `k=4096` rows; both
are much faster than `basic`, `shared_ab`, and `warp_const_b`.

## Best Scaling Results

From the heatmaps in [p4_skip_cpu.txt](p4_skip_cpu.txt):

- `scale_sq / basic`: best `0.982 ms` at `(256, 16)`
- `scale_sq / constant_b`: best `0.979 ms` at `(128, 128)`
- `scale_sq / shared_ab`: best `1.585 ms` at `(128, 128)`
- `scale_sq / warp`: best `0.365 ms` at `(128, 128)`
- `scale_sq / warp_const_b`: best `1.296 ms` at `(64, 64)`
- `scale_sq / warp_per_row`: best `0.362 ms` at `(128, 128)`
- `scale_tall / basic`: best `0.982 ms` at `(256, 16)`
- `scale_tall / constant_b`: best `0.980 ms` at `(128, 32)`
- `scale_tall / shared_ab`: best `1.584 ms` at `(128, 128)`
- `scale_tall / warp`: best `0.366 ms` at `(128, 128)`
- `scale_tall / warp_const_b`: best `1.226 ms` at `(64, 64)`
- `scale_tall / warp_per_row`: best `0.362 ms` at `(512, 32)`
- `scale_wide / basic`: best `1.044 ms` at `(64, 128)`
- `scale_wide / constant_b`: best `1.051 ms` at `(64, 128)`
- `scale_wide / shared_ab`: best `1.583 ms` at `(128, 128)`
- `scale_wide / warp`: best `0.362 ms` at `(128, 128)`
- `scale_wide / warp_const_b`: best `1.303 ms` at `(64, 32)`
- `scale_wide / warp_per_row`: best `0.362 ms` at `(256, 64)`

## Notes

- Complete row-by-row data lives in the raw `.txt` logs.
- The `warp_const_b` experiment is slower than `warp` because each warp lane
  reads a different `B` address. Constant memory is strongest for broadcast
  reads where all lanes read the same address; this warp layout instead turns
  a coalesced global load into a poor constant-memory access pattern.
- The one-warp-per-output-row mapping removes the shared-memory atomic used
  by the original `warp` kernel and remains the strongest overall direction.
