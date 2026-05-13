# `P4_MVM.cu` Results

Updated summary based on the current four-variant matrix-vector harness.

- CPU-backed correctness run: [p4_with_cpu.txt](p4_with_cpu.txt)
- Heavier benchmark run: [p4_skip_cpu.txt](p4_skip_cpu.txt)
- Follow-up ideas: [P4_MVM_OPTIMIZATION_NOTES.md](P4_MVM_OPTIMIZATION_NOTES.md)

## Kernel Variants

- `basic`: direct global-memory implementation
- `constant_b`: direct implementation with `B` in constant memory
- `shared_ab`: shared-memory tiles for `A` and `B`
- `warp`: register accumulation with warp-level reduction

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
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

- All four kernels pass the small exact tests.
- All four kernels pass the CPU-reference medium and large cases.
- The CPU-backed launch sweep passes for every tested variant.
- The refreshed CPU-backed log has `16 PASS/PASS`, `108 REF/PASS`,
  `0 FAIL`, and `0 SKIP` result rows.

From [p4_skip_cpu.txt](p4_skip_cpu.txt):

- The skip-CPU benchmark log has `16 SKIP/PASS`, `304 SKIP/SKIP`,
  and `0 FAIL` result rows.
- The small exact rows are still checked even in skip-CPU mode.
- Larger Tensara and shape rows are benchmark-only in this log.

## Performance Summary

From [p4_with_cpu.txt](p4_with_cpu.txt):

- `warp` wins 18 of 31 comparable CPU-backed configurations.
- `constant_b` wins 5 configurations.
- `basic` wins 4 configurations.
- `shared_ab` wins 3 configurations.
- There is 1 tie.

From [p4_skip_cpu.txt](p4_skip_cpu.txt):

- `warp` wins 65 of 80 comparable benchmark configurations.
- `constant_b` wins 11 configurations.
- `basic` wins 1 configuration.
- `shared_ab` wins 1 configuration.
- There are 2 ties.

The broad result is clear: `warp` is the best current kernel. `constant_b`
helps relative to `basic` on larger fixed-launch rows, while `shared_ab` is
correct but usually pays too much synchronization and shared-memory overhead.

## Tensara-Size Rows

From [p4_skip_cpu.txt](p4_skip_cpu.txt), default launch, `kernel_ms`:

```text
tensara_1 4096x4096  basic 1.487  constant_b 1.072  shared_ab 2.432  warp 0.365
tensara_2 6144x4096  basic 1.538  constant_b 1.529  shared_ab 3.641  warp 0.550
tensara_3 7168x4096  basic 1.845  constant_b 1.755  shared_ab 4.240  warp 0.635
tensara_4 8192x4096  basic 2.088  constant_b 1.967  shared_ab 4.844  warp 0.727
tensara_5 9216x4096  basic 2.309  constant_b 2.195  shared_ab 5.476  warp 0.814
```

The `warp` kernel is about `3x` to `6x` faster than `shared_ab` on these rows
and about `2.7x` to `4.1x` faster than the basic direct implementation.

## Best Scaling Results

From the heatmaps in [p4_skip_cpu.txt](p4_skip_cpu.txt):

- `scale_sq / basic`: best `0.981 ms` at `(128, 64)`
- `scale_sq / constant_b`: best `0.979 ms` at `(64, 64)`
- `scale_sq / shared_ab`: best `1.578 ms` at `(128, 128)`
- `scale_sq / warp`: best `0.367 ms` at `(256, 64)`
- `scale_tall / basic`: best `0.981 ms` at `(64, 64)`
- `scale_tall / constant_b`: best `0.979 ms` at `(256, 16)`
- `scale_tall / shared_ab`: best `1.579 ms` at `(128, 128)`
- `scale_tall / warp`: best `0.366 ms` at `(128, 128)`
- `scale_wide / basic`: best `1.070 ms` at `(64, 32)`
- `scale_wide / constant_b`: best `1.071 ms` at `(64, 64)`
- `scale_wide / shared_ab`: best `1.578 ms` at `(128, 128)`
- `scale_wide / warp`: best `0.364 ms` at `(128, 128)`

## Notes

- Complete row-by-row data lives in the raw `.txt` logs.
- The next optimization direction is not more shared memory for one row.
  The stronger path is likely multi-row-per-block work, starting with one
  warp per output row.
