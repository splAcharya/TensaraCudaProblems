# `P1_1D_CONVOLUTIONS.cu` Results

Detailed benchmark notes and raw output for the 1D convolution problem.

## Kernel Variants

- `basic`: direct global-memory implementation
- `tiled`: shared-memory tiled implementation with halo loads
- `bstride`: shared-memory tiled implementation with block-stride loading

## Local Benchmark Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- Default launch for the main benchmark rows: `block_x=256`, `grid_x=32`
- Scaling sweep launches:
  - `block_x in [32, 64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64]`

## Summary

- All three kernels pass the current correctness checks.
- `tiled` is correct but is generally the weakest performer for large `K`.
- `bstride` is the best shared-memory version so far.
- On the largest local convolution case in the default sweep,
  `tile_5 (N=2097152, K=511)`, `bstride` improves `kernel_ms` from `6.317`
  (`basic`) and `8.613` (`tiled`) down to `6.072`.
- On the website-style `K=8191` cases, `bstride` is best at the default launch:
  - `web_1`: `basic=1.605`, `tiled=1.936`, `bstride=1.521`
  - `web_2`: `basic=3.342`, `tiled=3.882`, `bstride=3.016`
- In the scaling sweep, the best launch shapes are usually in the
  `block_x=256..512` range with moderate-to-high `grid_x`.

## Best Scaling Results

Using `kernel_ms` as the main metric:

- `web_1 / basic`: best `1.112 ms` at `(block_x=512, grid_x=32)`
- `web_1 / tiled`: best `1.303 ms` at `(block_x=512, grid_x=64)`
- `web_1 / bstride`: best `1.127 ms` at `(block_x=512, grid_x=64)`
- `web_2 / basic`: best `2.194 ms` at `(block_x=512, grid_x=32)`
- `web_2 / tiled`: best `2.637 ms` at `(block_x=512, grid_x=32)`
- `web_2 / bstride`: best `2.203 ms` at `(block_x=256, grid_x=64)`

## Raw Benchmark Output

```text
group   name        kernel    N         K       block_x grid_x  cpu   gpu   total_ms    kernel_ms
----------------------------------------------------------------------------------------------------
small   small_1     basic     4         3       0       0       PASS  PASS  3.479       2.930
small   small_1     tiled     4         3       0       0       PASS  PASS  0.572       0.094
small   small_1     bstride   4         3       0       0       PASS  PASS  0.506       0.124
small   small_2     basic     3         3       0       0       PASS  PASS  0.482       0.026
small   small_2     tiled     3         3       0       0       PASS  PASS  0.397       0.025
small   small_2     bstride   3         3       0       0       PASS  PASS  0.394       0.026
small   small_3     basic     3         3       0       0       PASS  PASS  0.358       0.027
small   small_3     tiled     3         3       0       0       PASS  PASS  0.371       0.025
small   small_3     bstride   3         3       0       0       PASS  PASS  0.355       0.029
large   large_1     basic     32768     31      256     32      REF   PASS  0.502       0.030
large   large_1     tiled     32768     31      256     32      REF   PASS  0.656       0.106
large   large_1     bstride   32768     31      256     32      REF   PASS  0.570       0.053
large   large_2     basic     65536     63      256     32      REF   PASS  0.725       0.118
large   large_2     tiled     65536     63      256     32      REF   PASS  0.639       0.127
large   large_2     bstride   65536     63      256     32      REF   PASS  0.617       0.081
large   large_3     basic     131072    95      256     32      REF   PASS  0.641       0.101
large   large_3     tiled     131072    95      256     32      REF   PASS  3.309       0.123
large   large_3     bstride   131072    95      256     32      REF   PASS  0.814       0.100
large   large_4     basic     262144    127     256     32      REF   PASS  2.347       0.258
large   large_4     tiled     262144    127     256     32      REF   PASS  1.369       0.300
large   large_4     bstride   262144    127     256     32      REF   PASS  1.245       0.221
large   large_5     basic     524288    191     256     32      REF   PASS  3.729       0.653
large   large_5     tiled     524288    191     256     32      REF   PASS  2.182       0.810
large   large_5     bstride   524288    191     256     32      REF   PASS  2.121       0.648
tile    tile_1      basic     262144    127     256     32      REF   PASS  2.833       0.263
tile    tile_1      tiled     262144    127     256     32      REF   PASS  2.017       0.398
tile    tile_1      bstride   262144    127     256     32      REF   PASS  1.293       0.221
tile    tile_2      basic     524288    191     256     32      REF   PASS  3.916       0.729
tile    tile_2      tiled     524288    191     256     32      REF   PASS  2.393       0.808
tile    tile_2      bstride   524288    191     256     32      REF   PASS  2.138       0.654
tile    tile_3      basic     1048576   255     256     32      REF   PASS  4.249       1.618
tile    tile_3      tiled     1048576   255     256     32      REF   PASS  3.934       2.023
tile    tile_3      bstride   1048576   255     256     32      REF   PASS  3.629       1.591
tile    tile_4      basic     1048576   383     256     32      REF   PASS  5.190       2.428
tile    tile_4      tiled     1048576   383     256     32      REF   PASS  5.108       3.029
tile    tile_4      bstride   1048576   383     256     32      REF   PASS  4.464       2.346
tile    tile_5      basic     2097152   511     256     32      REF   PASS  10.829      6.317
tile    tile_5      tiled     2097152   511     256     32      REF   PASS  11.690      8.613
tile    tile_5      bstride   2097152   511     256     32      REF   PASS  10.247      6.072
web     web_1       basic     32768     8191    256     32      REF   PASS  2.495       1.605
web     web_1       tiled     32768     8191    256     32      REF   PASS  2.421       1.936
web     web_1       bstride   32768     8191    256     32      REF   PASS  1.979       1.521
web     web_2       basic     65536     8191    256     32      REF   PASS  4.904       3.342
web     web_2       tiled     65536     8191    256     32      REF   PASS  4.999       3.882
web     web_2       bstride   65536     8191    256     32      REF   PASS  3.828       3.016
```

## Scaling Tables

GitHub markdown tables do not support true cell background colors in a portable
way. These tables use colored square emoji instead:

- `🟩` best or near-best
- `🟨` close
- `🟧` moderate gap
- `🟥` clearly worse

### `web_1 / basic`

Best: `(block_x=512, grid_x=32)`, `1.112 ms`

| block_x \ grid_x | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| 32  | 🟥 17.280 | 🟥 8.873 | 🟥 4.683 | 🟥 2.501 |
| 64  | 🟥 6.651 | 🟥 3.514 | 🟥 1.940 | 🟨 1.333 |
| 128 | 🟥 3.627 | 🟥 2.042 | 🟨 1.368 | 🟩 1.194 |
| 256 | 🟥 2.484 | 🟧 1.452 | 🟩 1.155 | 🟩 1.170 |
| 512 | 🟥 2.192 | 🟩 1.128 | 🟩 1.112 | 🟩 1.123 |

### `web_1 / tiled`

Best: `(block_x=512, grid_x=64)`, `1.303 ms`

| block_x \ grid_x | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| 32  | 🟥 44.005 | 🟥 22.047 | 🟥 10.971 | 🟥 8.421 |
| 64  | 🟥 12.637 | 🟥 6.352 | 🟥 3.175 | 🟥 3.192 |
| 128 | 🟥 6.645 | 🟥 3.347 | 🟧 1.907 | 🟧 1.879 |
| 256 | 🟥 3.826 | 🟧 1.946 | 🟨 1.480 | 🟨 1.476 |
| 512 | 🟥 2.986 | 🟨 1.531 | 🟩 1.348 | 🟩 1.303 |

### `web_1 / bstride`

Best: `(block_x=512, grid_x=64)`, `1.127 ms`

| block_x \ grid_x | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| 32  | 🟥 20.739 | 🟥 10.388 | 🟥 5.155 | 🟥 3.927 |
| 64  | 🟥 6.707 | 🟥 3.375 | 🟥 1.942 | 🟥 1.959 |
| 128 | 🟥 3.654 | 🟥 1.867 | 🟨 1.254 | 🟩 1.222 |
| 256 | 🟥 2.273 | 🟩 1.165 | 🟩 1.145 | 🟩 1.136 |
| 512 | 🟥 2.214 | 🟩 1.138 | 🟩 1.134 | 🟩 1.127 |

### `web_2 / basic`

Best: `(block_x=512, grid_x=32)`, `2.194 ms`

| block_x \ grid_x | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| 32  | 🟥 26.082 | 🟥 13.439 | 🟥 6.947 | 🟥 3.664 |
| 64  | 🟥 13.348 | 🟥 7.123 | 🟥 3.832 | 🟨 2.635 |
| 128 | 🟥 7.226 | 🟥 4.053 | 🟨 2.645 | 🟨 2.553 |
| 256 | 🟥 4.939 | 🟧 2.848 | 🟨 2.454 | 🟩 2.252 |
| 512 | 🟥 4.407 | 🟩 2.201 | 🟩 2.194 | 🟩 2.205 |

### `web_2 / tiled`

Best: `(block_x=512, grid_x=32)`, `2.637 ms`

| block_x \ grid_x | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| 32  | 🟥 66.786 | 🟥 33.496 | 🟥 16.622 | 🟥 15.852 |
| 64  | 🟥 25.668 | 🟥 12.827 | 🟥 6.385 | 🟥 6.342 |
| 128 | 🟥 13.510 | 🟥 6.829 | 🟧 3.890 | 🟧 3.793 |
| 256 | 🟥 7.666 | 🟧 3.917 | 🟨 2.916 | 🟩 2.866 |
| 512 | 🟥 5.936 | 🟨 3.082 | 🟩 2.637 | 🟩 2.659 |

### `web_2 / bstride`

Best: `(block_x=256, grid_x=64)`, `2.203 ms`

| block_x \ grid_x | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| 32  | 🟥 31.518 | 🟥 15.697 | 🟥 7.801 | 🟥 7.926 |
| 64  | 🟥 13.542 | 🟥 6.873 | 🟥 3.800 | 🟥 3.891 |
| 128 | 🟥 7.352 | 🟥 3.725 | 🟨 2.459 | 🟨 2.461 |
| 256 | 🟥 4.497 | 🟩 2.274 | 🟩 2.238 | 🟩 2.203 |
| 512 | 🟥 4.395 | 🟩 2.220 | 🟩 2.242 | 🟩 2.280 |
