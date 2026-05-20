# P4 Matrix Vector Multiplication Optimization Notes

## Current State

- Current kernels: `basic`, `constant_b`, `shared_ab`, `warp`,
  `warp_const_b`, and `warp_per_row`.
- Latest local logs:
  - `p4_with_cpu.txt`
  - `p4_skip_cpu.txt`
- The `warp_per_row` kernel is currently the strongest performer in the
  skip-CPU log.
- The original `warp` kernel remains very close on Tensara-size rows.
- Tensara submission improved from about `1.27 ms` to about `0.45 ms`.

## Follow-Up Ideas

### Reuse Vector B More Effectively

`B` is reused by every output row, so reducing repeated global reads is a
major optimization direction.

Ideas:

- Use constant memory when `k` fits and access patterns are favorable.
- Use read-only cache loads for `B`.
- Load tiles of `B` once per block and apply them to multiple rows.
- Compare whether explicit shared-memory tiling beats cache behavior.

Note: `__ldg()` read-only cache loads were tested for global `A`/`B` loads and
did not improve the kernels that matter most. `warp_per_row` was flat to
slightly slower, and `warp` slowed down. Keep this off unless a future GPU or
shape changes the result.

### Future: Shared `B` Tile For `warp_per_row`

The `warp_per_row` mapping has several warps per block. With `block_x=256`,
one block has 8 warps, and each warp computes a different output row. For the
same column tile, those warps all read the same `B` values.

Future experiment:

```text
block threads cooperatively load B[tile] into shared memory
__syncthreads()
each row-warp accumulates A[row, tile] * shared_B[tile]
__syncthreads()
repeat for the next tile
```

This could reduce repeated global/cache reads of `B` across warps in the same
block. The risk is extra shared-memory traffic and block-level synchronization.
It is worth exploring later, but there is no stub in `P4_MVM.cu` yet.

### Constant Memory Negative Result

The `warp_const_b` experiment keeps the `warp` mapping but reads `B` from
constant memory. It is slower than `warp` because each lane reads a different
`B` address:

```text
lane 0 -> B[col + 0]
lane 1 -> B[col + 1]
...
lane 31 -> B[col + 31]
```

That is a coalesced global-memory pattern, but not a constant-memory broadcast
pattern. Constant memory is still useful for row-per-thread mappings where all
lanes read the same `B[col]`, but it is not a promising direction for the
current warp-level layout. Keep `warp_const_b` as a display/negative-result
variant.

### Compute Multiple Rows Per Block

Current row ownership mostly centers around one output row per block. A
different mapping could let one block produce several output values.

Sketch:

```text
block owns rows r0..rN
load one tile of B
each row accumulates its own partial dot product
reduce per row
write N output values
```

This may improve reuse of `B` and increase useful work per block.

### Try One Warp Per Output Row

Assign each warp to one output row instead of using the whole block for one
row.

Sketch:

```text
block = several warps
each warp owns one row
lanes walk across k with col += 32
lane 0 writes the output
```

This removes the need for cross-warp shared-memory accumulation for one row.

Status: implemented as `warp_per_row`, verified by the CPU-backed harness,
and best on most skip-CPU benchmark configurations.

### Sweep Rows Per Block And Warps Per Row

Useful variants to benchmark:

```text
1 warp per row
2 warps per row
4 warps per row
8 warps per row
```

The best choice depends on `m`, `k`, memory bandwidth, and reduction overhead.

### Specialize For Tensara Shapes

If the target cases mostly use `k = 4096`, add a specialized fast path.

Possible simplifications:

- Remove tail checks when `k` is divisible by the tile size.
- Template block size so the compiler can simplify loops.
- Unroll the inner loop by a fixed factor.
- Tune the default launch config around the known benchmark shapes.

### Split Long Rows Across Blocks

For very large `k`, multiple blocks could compute partial sums for one row and
combine them with a second kernel or atomics.

This adds global accumulation overhead, so it is most likely useful when `k` is
large and `m` is too small to expose enough row parallelism.

### Reduce Synchronization And Atomics

Keep partial sums in registers as long as possible. Prefer reductions in this
order:

```text
register accumulation
warp shuffle
shared memory
global atomic
```

Shared memory is worth using only when the reuse it enables is greater than the
cost of synchronization and extra instructions.

## Next Session Starting Point

Start with a multi-row-per-block experiment. The simplest first variant is
one warp per output row, with several rows owned by one block. Compare that
against the current `warp` kernel on the existing `scale_*` and Tensara-size
benchmark rows.
