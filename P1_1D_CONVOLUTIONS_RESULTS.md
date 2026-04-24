# `P1_1D_CONVOLUTIONS.cu` Results

Detailed benchmark notes and raw output for the 1D convolution problem,
including the constant-memory comparison for filter `B`.

## Kernel Variants

- `basic`: direct global-memory implementation
- `basic_c`: direct implementation with `B` in constant memory
- `tiled`: shared-memory tiled implementation with halo loads
- `tiled_c`: tiled implementation with `B` in constant memory
- `bstride`: shared-memory tiled implementation with block-stride loading
- `bstride_c`: block-stride tiled implementation with `B` in constant memory

## Local Benchmark Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4 GB
- Default launch for the main benchmark rows: `block_x=256`, `grid_x=32`
- Scaling sweep launches:
  - `block_x in [32, 64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64]`

## Summary

- All six benchmarked variants pass the current correctness checks.
- Moving `B` to constant memory improves `basic` modestly and improves
  `tiled` and `bstride` substantially.
- At the default launch on the `K=8191` web-style cases:
  - `web_1`: `basic 6.017 -> basic_c 5.475`, `tiled 7.977 -> tiled_c 4.640`,
    `bstride 5.806 -> bstride_c 2.980`
  - `web_2`: `basic 22.087 -> basic_c 19.028`, `tiled 27.081 -> tiled_c 16.241`,
    `bstride 20.464 -> bstride_c 10.331`
- `bstride_c` is the current best implementation on the web-style scaling sweep.
- On the largest local tiled case at the default launch,
  `tile_5 (N=2097152, K=511)`, `bstride_c` reduces `kernel_ms` from `25.353`
  (`bstride`) and `55.277` (`tiled`) down to `13.293`.
- `basic_c` is not uniformly faster at every weak launch shape, but it wins or
  roughly ties the tuned launch points.

## Why Constant Memory Helps Here

The inner loop of this convolution has an access pattern that is unusually
friendly to constant memory:

```text
At loop step h, every lane in the warp reads the same filter element:

lane 0 : sum += A[i0 + h - r] * B[h]
lane 1 : sum += A[i1 + h - r] * B[h]
lane 2 : sum += A[i2 + h - r] * B[h]
...
lane 31: sum += A[i31 + h - r] * B[h]

A[...] changes per thread
B[h]   is the same for every thread in the warp at that moment
```

That means `B` is a good broadcast candidate:

```text
Without constant memory:
  32 lanes in a warp ask for B[h]
  -> the loads go through normal global-memory/cache paths

With constant memory:
  32 lanes in a warp ask for the same B[h]
  -> the constant cache can serve that value as a broadcast
```

Another way to view it:

```text
Warp at h = 123

            A index read                  Filter read
lane 0  ->  A[gx0 + 123 - r]         *   B[123]
lane 1  ->  A[gx1 + 123 - r]         *   B[123]
lane 2  ->  A[gx2 + 123 - r]         *   B[123]
...
lane31  ->  A[gx31 + 123 - r]        *   B[123]

Shared memory helps the A-side reuse.
Constant memory helps the B-side reuse.
```

This also explains why the biggest gains show up for `tiled_c` and
`bstride_c`: those kernels already reduce the cost of fetching `A`, so the
filter loads become a larger fraction of the remaining work. Once `A` is
better staged, optimizing `B` matters more.

## Constant Memory Limits

- Constant memory is small. In CUDA it is a small read-only space, commonly
  limited to 64 KB. This benchmark code reserves `8192` floats for `B`, which
  is `8192 * 4 = 32768` bytes, or 32 KB.
- It helps most when threads in a warp read the same address at the same time.
  That is true for `B[h]` here, but it is not true for arbitrary per-thread
  data.
- If lanes read many different constant-memory addresses in the same
  instruction, the access loses the clean broadcast behavior and can serialize.
- It is not a replacement for shared memory or global memory. It is best for
  small, read-only data that is reused broadly across threads.
- It does not scale to arbitrarily large filters. In this implementation the
  constant-memory comparison path is capped at `K <= 8192`, and the largest
  tested web-style cases use `K = 8191`, which still fits.
- There is still a host-side copy into constant memory before launch, so the
  technique is most useful when that setup cost is amortized by enough work in
  the kernel.

## Best Scaling Results

Using `kernel_ms` as the main metric:

- `web_1 / basic`: best `1.096 ms` at `(block_x=512, grid_x=32)`
- `web_1 / basic_c`: best `0.978 ms` at `(block_x=512, grid_x=64)`
- `web_1 / tiled`: best `1.310 ms` at `(block_x=512, grid_x=64)`
- `web_1 / tiled_c`: best `0.771 ms` at `(block_x=512, grid_x=64)`
- `web_1 / bstride`: best `1.136 ms` at `(block_x=512, grid_x=32)`
- `web_1 / bstride_c`: best `0.583 ms` at `(block_x=256, grid_x=64)`
- `web_2 / basic`: best `2.199 ms` at `(block_x=512, grid_x=16)`
- `web_2 / basic_c`: best `2.044 ms` at `(block_x=512, grid_x=64)`
- `web_2 / tiled`: best `2.599 ms` at `(block_x=512, grid_x=64)`
- `web_2 / tiled_c`: best `1.527 ms` at `(block_x=512, grid_x=64)`
- `web_2 / bstride`: best `2.186 ms` at `(block_x=512, grid_x=64)`
- `web_2 / bstride_c`: best `1.119 ms` at `(block_x=512, grid_x=16)`

## Raw Benchmark Output

```text
group   name        kernel    N         K       block_x grid_x  cpu   gpu   total_ms    kernel_ms
----------------------------------------------------------------------------------------------------
small   small_1     basic     4         3       256     32      PASS  PASS  6.386       5.814
small   small_1     basic_c   4         3       256     32      PASS  PASS  0.589       0.155
small   small_1     tiled     4         3       256     32      PASS  PASS  0.739       0.160
small   small_1     tiled_c   4         3       256     32      PASS  PASS  0.831       0.151
small   small_1     bstride   4         3       256     32      PASS  PASS  0.589       0.151
small   small_1     bstride_c 4         3       256     32      PASS  PASS  0.591       0.153
small   small_2     basic     3         3       256     32      PASS  PASS  0.529       0.105
small   small_2     basic_c   3         3       256     32      PASS  PASS  0.528       0.105
small   small_2     tiled     3         3       256     32      PASS  PASS  0.526       0.104
small   small_2     tiled_c   3         3       256     32      PASS  PASS  0.525       0.104
small   small_2     bstride   3         3       256     32      PASS  PASS  0.524       0.105
small   small_2     bstride_c 3         3       256     32      PASS  PASS  0.524       0.104
small   small_3     basic     3         3       256     32      PASS  PASS  0.535       0.103
small   small_3     basic_c   3         3       256     32      PASS  PASS  0.517       0.050
small   small_3     tiled     3         3       256     32      PASS  PASS  0.522       0.096
small   small_3     tiled_c   3         3       256     32      PASS  PASS  0.504       0.051
small   small_3     bstride   3         3       256     32      PASS  PASS  0.532       0.100
small   small_3     bstride_c 3         3       256     32      PASS  PASS  0.531       0.103
large   large_1     basic     32768     31      256     32      REF   PASS  0.810       0.114
large   large_1     basic_c   32768     31      256     32      REF   PASS  0.585       0.032
large   large_1     tiled     32768     31      256     32      REF   PASS  0.696       0.100
large   large_1     tiled_c   32768     31      256     32      REF   PASS  0.634       0.093
large   large_1     bstride   32768     31      256     32      REF   PASS  0.635       0.051
large   large_1     bstride_c 32768     31      256     32      REF   PASS  0.595       0.052
large   large_2     basic     65536     63      256     32      REF   PASS  1.307       0.132
large   large_2     basic_c   65536     63      256     32      REF   PASS  0.824       0.046
large   large_2     tiled     65536     63      256     32      REF   PASS  1.012       0.189
large   large_2     tiled_c   65536     63      256     32      REF   PASS  0.815       0.045
large   large_2     bstride   65536     63      256     32      REF   PASS  0.690       0.120
large   large_2     bstride_c 65536     63      256     32      REF   PASS  0.655       0.083
large   large_3     basic     131072    95      256     32      REF   PASS  1.853       0.159
large   large_3     basic_c   131072    95      256     32      REF   PASS  0.955       0.091
large   large_3     tiled     131072    95      256     32      REF   PASS  1.154       0.163
large   large_3     tiled_c   131072    95      256     32      REF   PASS  1.467       0.181
large   large_3     bstride   131072    95      256     32      REF   PASS  0.808       0.127
large   large_3     bstride_c 131072    95      256     32      REF   PASS  0.760       0.094
large   large_4     basic     262144    127     256     32      REF   PASS  3.098       0.286
large   large_4     basic_c   262144    127     256     32      REF   PASS  1.210       0.202
large   large_4     tiled     262144    127     256     32      REF   PASS  2.133       0.374
large   large_4     tiled_c   262144    127     256     32      REF   PASS  1.289       0.233
large   large_4     bstride   262144    127     256     32      REF   PASS  1.268       0.234
large   large_4     bstride_c 262144    127     256     32      REF   PASS  1.075       0.154
large   large_5     basic     524288    191     256     32      REF   PASS  3.743       0.683
large   large_5     basic_c   524288    191     256     32      REF   PASS  2.030       0.600
large   large_5     tiled     524288    191     256     32      REF   PASS  2.398       0.847
large   large_5     tiled_c   524288    191     256     32      REF   PASS  2.180       0.534
large   large_5     bstride   524288    191     256     32      REF   PASS  2.818       0.663
large   large_5     bstride_c 524288    191     256     32      REF   PASS  1.861       0.377
tile    tile_1      basic     262144    127     256     32      REF   PASS  2.252       0.287
tile    tile_1      basic_c   262144    127     256     32      REF   PASS  1.148       0.205
tile    tile_1      tiled     262144    127     256     32      REF   PASS  1.677       0.299
tile    tile_1      tiled_c   262144    127     256     32      REF   PASS  1.452       0.244
tile    tile_1      bstride   262144    127     256     32      REF   PASS  1.320       0.245
tile    tile_1      bstride_c 262144    127     256     32      REF   PASS  1.173       0.220
tile    tile_2      basic     524288    191     256     32      REF   PASS  3.127       0.634
tile    tile_2      basic_c   524288    191     256     32      REF   PASS  2.720       0.598
tile    tile_2      tiled     524288    191     256     32      REF   PASS  3.330       0.806
tile    tile_2      tiled_c   524288    191     256     32      REF   PASS  2.240       0.643
tile    tile_2      bstride   524288    191     256     32      REF   PASS  2.134       0.621
tile    tile_2      bstride_c 524288    191     256     32      REF   PASS  1.842       0.343
tile    tile_3      basic     1048576   255     256     32      REF   PASS  4.380       1.614
tile    tile_3      basic_c   1048576   255     256     32      REF   PASS  3.811       1.498
tile    tile_3      tiled     1048576   255     256     32      REF   PASS  4.853       1.975
tile    tile_3      tiled_c   1048576   255     256     32      REF   PASS  4.369       1.436
tile    tile_3      bstride   1048576   255     256     32      REF   PASS  4.602       1.594
tile    tile_3      bstride_c 1048576   255     256     32      REF   PASS  3.707       0.864
tile    tile_4      basic     1048576   383     256     32      REF   PASS  6.151       2.399
tile    tile_4      basic_c   1048576   383     256     32      REF   PASS  4.701       2.247
tile    tile_4      tiled     1048576   383     256     32      REF   PASS  5.563       2.995
tile    tile_4      tiled_c   1048576   383     256     32      REF   PASS  4.064       1.991
tile    tile_4      bstride   1048576   383     256     32      REF   PASS  4.606       2.352
tile    tile_4      bstride_c 1048576   383     256     32      REF   PASS  3.998       1.249
tile    tile_5      basic     2097152   511     256     32      REF   PASS  57.767      43.184
tile    tile_5      basic_c   2097152   511     256     32      REF   PASS  54.339      39.221
tile    tile_5      tiled     2097152   511     256     32      REF   PASS  83.386      55.277
tile    tile_5      tiled_c   2097152   511     256     32      REF   PASS  40.922      26.715
tile    tile_5      bstride   2097152   511     256     32      REF   PASS  39.546      25.353
tile    tile_5      bstride_c 2097152   511     256     32      REF   PASS  29.118      13.293
web     web_1       basic     32768     8191    256     32      REF   PASS  7.572       6.017
web     web_1       basic_c   32768     8191    256     32      REF   PASS  6.911       5.475
web     web_1       tiled     32768     8191    256     32      REF   PASS  21.479      7.977
web     web_1       tiled_c   32768     8191    256     32      REF   PASS  6.261       4.640
web     web_1       bstride   32768     8191    256     32      REF   PASS  7.332       5.806
web     web_1       bstride_c 32768     8191    256     32      REF   PASS  4.696       2.980
web     web_2       basic     65536     8191    256     32      REF   PASS  23.477      22.087
web     web_2       basic_c   65536     8191    256     32      REF   PASS  20.388      19.028
web     web_2       tiled     65536     8191    256     32      REF   PASS  29.021      27.081
web     web_2       tiled_c   65536     8191    256     32      REF   PASS  17.590      16.241
web     web_2       bstride   65536     8191    256     32      REF   PASS  21.866      20.464
web     web_2       bstride_c 65536     8191    256     32      REF   PASS  11.834      10.331
```

## Scaling Heatmaps

These are the direct console heatmaps from the comparison run, using
`kernel_ms` as the metric and lower-is-better semantics.

```text
web_1 / basic  best=(512,32) 1.096 ms
block\grid8         16        32        64
32        80.805    8.954     3.579     1.878
64        6.777     3.550     1.945     1.342
128       3.658     2.026     1.381     1.198
256       2.495     1.462     1.191     1.151
512       2.195     1.129     1.096     1.124

web_1 / basic_c  best=(512,64) 0.978 ms
block\grid8         16        32        64
32        86.203    10.994    4.254     2.233
64        8.223     4.258     2.326     1.423
128       4.404     2.355     1.421     1.072
256       2.771     1.623     1.106     1.080
512       2.052     1.086     1.029     0.978

web_1 / tiled  best=(512,64) 1.310 ms
block\grid8         16        32        64
32        44.180    22.126    8.276     7.914
64        12.825    6.381     3.247     3.184
128       6.691     3.365     1.931     1.917
256       3.811     1.939     1.496     1.461
512       2.982     1.512     1.453     1.310

web_1 / tiled_c  best=(512,64) 0.771 ms
block\grid8         16        32        64
32        33.950    17.095    6.331     6.198
64        8.989     4.477     2.308     2.342
128       4.483     2.270     1.412     1.392
256       2.441     1.347     0.926     0.881
512       1.864     1.063     0.813     0.771

web_1 / bstride  best=(512,32) 1.136 ms
block\grid8         16        32        64
32        20.898    7.908     3.927     3.967
64        6.744     3.382     1.926     2.031
128       3.674     1.869     1.255     1.242
256       2.274     1.166     1.157     1.154
512       2.222     1.141     1.136     1.139

web_1 / bstride_c  best=(256,64) 0.583 ms
block\grid8         16        32        64
32        12.838    4.853     2.438     2.356
64        3.777     1.892     1.123     1.012
128       2.033     1.024     0.658     0.673
256       1.267     0.634     0.687     0.583
512       1.142     0.601     0.596     0.602

web_2 / basic  best=(512,16) 2.199 ms
block\grid8         16        32        64
32        138.171   13.344    6.993     3.687
64        13.434    7.007     3.784     2.641
128       7.159     3.977     2.646     2.638
256       4.894     2.841     2.586     2.266
512       4.328     2.199     2.202     2.234

web_2 / basic_c  best=(512,64) 2.044 ms
block\grid8         16        32        64
32        110.102   16.521    8.532     4.434
64        16.438    8.460     4.506     2.895
128       8.784     4.687     2.786     2.107
256       5.421     3.202     2.247     2.265
512       3.973     2.194     2.315     2.044

web_2 / tiled  best=(512,64) 2.599 ms
block\grid8         16        32        64
32        88.678    33.493    16.713    15.875
64        25.630    12.795    6.393     6.370
128       13.411    6.761     3.804     3.769
256       7.597     3.811     2.931     2.895
512       5.945     3.006     2.655     2.599

web_2 / tiled_c  best=(512,64) 1.527 ms
block\grid8         16        32        64
32        68.713    25.937    12.848    12.326
64        18.166    8.986     4.743     4.589
128       9.051     4.558     2.758     2.716
256       4.994     2.695     1.887     1.779
512       3.768     1.913     1.573     1.527

web_2 / bstride  best=(512,64) 2.186 ms
block\grid8         16        32        64
32        34.566    15.752    7.815     7.958
64        13.533    6.857     3.859     3.848
128       7.341     3.741     2.502     2.435
256       4.477     2.277     2.210     2.217
512       4.393     2.232     2.234     2.186

web_2 / bstride_c  best=(512,16) 1.119 ms
block\grid8         16        32        64
32        19.605    9.830     4.924     4.714
64        7.658     3.840     2.256     2.032
128       4.151     1.990     1.254     1.261
256       2.370     1.172     1.148     1.162
512       2.299     1.119     1.120     1.165
```
