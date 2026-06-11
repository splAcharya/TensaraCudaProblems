#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Tensara-style signature:
// - input_a, input_b, output_c are device pointers
// - input_a is a row-major matrix with shape (m, k)
// - input_b is a vector with shape (k)
// - output_c is a vector with shape (m)
extern "C" void solution(const float* input_a, const float* input_b,
                         float* output_c, size_t m, size_t k);

// CPU reference is available for correctness verification.
static constexpr bool kCpuReferenceImplemented = true;
static constexpr size_t kConstantInputBElements = 8192;
static constexpr int kDefaultTimingRepeats = 5;
static constexpr int kTimingWarmupIterations = 1;
static constexpr int kProfileWarmupIterations = 5;
static constexpr int kProfileIterations = 50;

__constant__ float g_input_b_constant[kConstantInputBElements];

struct Timing {
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

struct LaunchConfig {
  int block_x = 256;
  int grid_x = 64;
};

enum class KernelVariant {
  kBasic,
  kConstantB,
  kSharedAB,
  kWarp,
  kWarpConstB,
  kWarpPerRow,
};

enum class TimingMode {
  kMedian,
  kBest,
};

static LaunchConfig g_launch_config{256, 64};
static KernelVariant g_kernel_variant = KernelVariant::kBasic;
static TimingMode g_timing_mode = TimingMode::kMedian;
static int g_timing_repeats = kDefaultTimingRepeats;
static bool g_kernel_arg_set = false;
static Timing g_last_timing;

static const char* current_kernel_name() {
  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      return "basic";
    case KernelVariant::kConstantB:
      return "constant_b";
    case KernelVariant::kSharedAB:
      return "shared_ab";
    case KernelVariant::kWarp:
      return "warp";
    case KernelVariant::kWarpConstB:
      return "warp_const_b";
    case KernelVariant::kWarpPerRow:
      return "warp_per_row";
  }
  return "unknown";
}

static bool kernel_enabled(KernelVariant variant) {
  return !g_kernel_arg_set || g_kernel_variant == variant;
}

static bool uses_constant_b(KernelVariant variant) {
  return variant == KernelVariant::kConstantB ||
         variant == KernelVariant::kWarpConstB;
}

static const char* timing_mode_name() {
  switch (g_timing_mode) {
    case TimingMode::kMedian:
      return "median";
    case TimingMode::kBest:
      return "best";
  }
  return "unknown";
}

static float select_timing_sample(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  if (g_timing_mode == TimingMode::kBest) {
    return samples.front();
  }
  return samples[samples.size() / 2];
}

static bool parse_kernel_arg(const std::string& arg) {
  const std::string prefix = "--kernel=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }

  const std::string value = arg.substr(prefix.size());
  if (value == "basic") {
    g_kernel_variant = KernelVariant::kBasic;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "constant-b" || value == "constant_b") {
    g_kernel_variant = KernelVariant::kConstantB;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "shared-ab" || value == "shared_ab") {
    g_kernel_variant = KernelVariant::kSharedAB;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "warp") {
    g_kernel_variant = KernelVariant::kWarp;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "warp-const-b" || value == "warp_const_b" ||
      value == "warp-constant-b") {
    g_kernel_variant = KernelVariant::kWarpConstB;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "warp-per-row" || value == "warp_per_row" ||
      value == "warp-row") {
    g_kernel_variant = KernelVariant::kWarpPerRow;
    g_kernel_arg_set = true;
    return true;
  }

  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic, --kernel=constant-b, "
            << "--kernel=shared-ab, --kernel=warp, "
            << "--kernel=warp-const-b, or --kernel=warp-per-row)\n";
  return false;
}

static bool parse_timing_arg(const std::string& arg) {
  const std::string prefix = "--timing=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }

  const std::string value = arg.substr(prefix.size());
  if (value == "median") {
    g_timing_mode = TimingMode::kMedian;
    return true;
  }
  if (value == "best" || value == "min") {
    g_timing_mode = TimingMode::kBest;
    return true;
  }

  std::cerr << "Unknown timing mode: " << value
            << " (use --timing=median or --timing=best)\n";
  return false;
}

static bool parse_timing_repeats_arg(const std::string& arg) {
  const std::string prefix = "--timing-repeats=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }

  const std::string value = arg.substr(prefix.size());
  try {
    size_t end = 0;
    const int repeats = std::stoi(value, &end);
    if (end != value.size() || repeats < 1) {
      throw std::invalid_argument("bad repeats");
    }
    g_timing_repeats = repeats;
    return true;
  } catch (const std::exception&) {
    std::cerr << "Invalid timing repeat count: " << value
              << " (use an integer >= 1)\n";
    return false;
  }
}

static bool cuda_runtime_ready() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    std::cerr << "CUDA runtime unavailable: " << cudaGetErrorString(err)
              << '\n';
    return false;
  }
  if (device_count <= 0) {
    std::cerr << "CUDA runtime unavailable: no CUDA devices found\n";
    return false;
  }
  return true;
}

inline void cuda_check(cudaError_t err, const char* file, int line,
                       const char* expr) {
  if (err == cudaSuccess) {
    return;
  }
  std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << expr
            << ") at " << file << ":" << line << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__, #expr)

struct TestCase {
  const char* name = "";
  size_t m = 0;
  size_t k = 0;
  std::vector<float> input_a;
  std::vector<float> input_b;
  std::vector<float> expected;
};

// CPU reference stub.
//
// input_a: row-major matrix with shape (m, k), stored as
//          input_a[row * k + col]
// input_b: vector with shape (k)
// output_c: vector with shape (m), where output_c[row] is the row dot product
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
static void cpu_matrix_vector(
    const std::vector<float>& input_a,
    const std::vector<float>& input_b,
    std::vector<float>& output_c,
    size_t m,
    size_t k)
{
  for (size_t row = 0; row < m; ++row)
  {
    float rsum = 0.0f;

    for (size_t col = 0; col < k; ++col)
      rsum += (input_a[row * k + col] * input_b[col]);

    output_c[row] = rsum;
  }
}

// Basic GPU kernel.
//
// input_a: device pointer to a row-major matrix with shape (m, k),
//          stored as input_a[row * k + col]
// input_b: device pointer to a vector with shape (k)
// output_c: device pointer to a vector with shape (m)
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
__global__ void device_mvm_basic(const float* input_a,
                                 const float* input_b,
                                 float* output_c, size_t m, size_t k) {
  const size_t gix =
      static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const size_t grid_stride =
      static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t row = gix; row < m; row += grid_stride) {
    float rsum = 0.0f;

    for (size_t col = 0; col < k; ++col) {
      rsum += input_a[row * k + col] * input_b[col];
    }

    output_c[row] = rsum;
  }
}

// Constant-memory input_b GPU kernel.
//
// input_a: device pointer to a row-major matrix with shape (m, k),
//          stored as input_a[row * k + col]
// output_c: device pointer to a vector with shape (m)
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
__global__ void device_mvm_constant_b(const float* input_a, float* output_c,
                                      size_t m, size_t k) {
  const size_t gix =
      static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const size_t grid_stride =
      static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t row = gix; row < m; row += grid_stride) {
    float rsum = 0.0f;

    for (size_t col = 0; col < k; ++col) {
      rsum += input_a[row * k + col] * g_input_b_constant[col];
    }

    output_c[row] = rsum;
  }
}

// Shared-memory input_a/input_b GPU kernel.
//
// input_a: device pointer to a row-major matrix with shape (m, k),
//          stored as input_a[row * k + col]
// input_b: device pointer to a vector with shape (k)
// output_c: device pointer to a vector with shape (m)
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
// shared memory: first blockDim.x floats are intended for input_a values;
//                next blockDim.x floats are intended for input_b values
__global__ void device_mvm_shared_ab(const float* input_a,
                                     const float* input_b, float* output_c,
                                     size_t m, size_t k) {

  //high level
  // each output tile is assigned a block of threads
  // so technically this is a block stride
  // how many blocks required for all outputs

  size_t total_blocks = m;
  size_t block_stride = gridDim.x;

  // Dynamic shared memory is one per-block slab. Two separate
  // extern __shared__ arrays would alias the same base address, so carve the
  // slab manually: A uses [0, blockDim.x), B uses [blockDim.x, 2*blockDim.x).
  // This keeps the two loaded tiles from overwriting each other before the
  // per-block reduction reads them.
  extern __shared__ float smem[];
  float *tile_a = smem;
  float *tile_b = tile_a + blockDim.x;

  size_t num_tiles =  (k + blockDim.x - 1) / blockDim.x;

  //each output cell is assiged a block
  for (size_t bx = blockIdx.x; bx < total_blocks; bx += block_stride)
  {
    //load tiles from gmem to smem
    float rsum = 0.0f;
    for (size_t tile_id = 0; tile_id < num_tiles; ++tile_id)
    {
      //load current a tile

      //each output row is assigned a block, i.e
      //each block is responsible for a row in input_a
      size_t row_pos_a = bx;

      //we bring in blockdim worth of elements per row into the current tile
      //within each tile, each thread will load a particular element
      size_t col_pos_a = (tile_id * blockDim.x) + threadIdx.x;
      size_t load_idx_a = (row_pos_a * k) + col_pos_a;
      tile_a[threadIdx.x] =
          (row_pos_a < m && col_pos_a < k) ? input_a[load_idx_a] : 0.0f;

      //load tile b
      size_t col_pos_b = (tile_id * blockDim.x) + threadIdx.x;
      tile_b[threadIdx.x] =
          (col_pos_b < k) ? input_b[col_pos_b] : 0.0f;

      //wait for the entire block to finish writting to shared memory
      __syncthreads();

      //only 1 thread per block needs to compute the sum,
      if (threadIdx.x == 0)
      {
        for (size_t rdim = 0; rdim < blockDim.x; ++rdim)
          rsum +=  tile_a[rdim] * tile_b[rdim];
      }

      //wait for shared memory reads to complete
      __syncthreads();
    }

    //store output
    if (threadIdx.x == 0 && bx < m)
      output_c[bx] = rsum;
  }
}

// Warp-level input_a/input_b GPU kernel.
//
// input_a: device pointer to a row-major matrix with shape (m, k),
//          stored as input_a[row * k + col]
// input_b: device pointer to a vector with shape (k)
// output_c: device pointer to a vector with shape (m)
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
__global__ void device_mvm_warp(const float* input_a,
                                const float* input_b, float* output_c,
                                size_t m, size_t k) {

  size_t total_blocks = m; //each block is responsible for 1 output
  size_t grid_stride = gridDim.x; //at block level
  size_t num_tiles = (k + blockDim.x - 1) / blockDim.x;
  __shared__ float block_res;
  size_t lane_id = threadIdx.x % 32;

  block_res = 0.0f;
  __syncthreads();

  for (size_t bx = blockIdx.x; bx < total_blocks; bx += grid_stride)
  {
    float rsum = 0.0f;

    //each blockdim worth of data will be operated on
    for (size_t tile_id = 0; tile_id < num_tiles; ++tile_id)
    {
      //load from global memory for local registers
      int row_pos_a = bx;
      int col_pos_a = (tile_id * blockDim.x) + threadIdx.x;
      int load_pos_a = (row_pos_a * k) + col_pos_a;
      float reg_a =
          (row_pos_a < m && col_pos_a < k) ? input_a[load_pos_a] : 0.0f;

      int load_pos_b = (tile_id * blockDim.x) + threadIdx.x;
      float reg_b = (load_pos_b < k) ? input_b[load_pos_b] : 0.0f;
      rsum += reg_a * reg_b;
    }

    //warp shuffle
    for (size_t lane_offset = 16; lane_offset > 0; lane_offset /= 2)
      rsum += __shfl_down_sync(0xffffffffu, rsum, lane_offset);

    if (lane_id == 0)
      atomicAdd(&block_res, rsum);

    __syncthreads();

    if (threadIdx.x == 0 && bx < m)
    {
      output_c[bx] = block_res;
      block_res = 0.0f;
    }

    __syncthreads();
  }
}

// Warp-level input_a/constant input_b GPU kernel.
//
// input_a: device pointer to a row-major matrix with shape (m, k),
//          stored as input_a[row * k + col]
// output_c: device pointer to a vector with shape (m)
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
__global__ void device_mvm_warp_const_b(const float* input_a,
                                        float* output_c, size_t m,
                                        size_t k) {
  size_t total_blocks = m;
  size_t grid_stride = gridDim.x;
  size_t num_tiles = (k + blockDim.x - 1) / blockDim.x;
  __shared__ float block_res;
  size_t lane_id = threadIdx.x % 32;

  block_res = 0.0f;
  __syncthreads();

  for (size_t bx = blockIdx.x; bx < total_blocks; bx += grid_stride) {
    float rsum = 0.0f;

    for (size_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
      const int row_pos_a = static_cast<int>(bx);
      const int col_pos_a =
          static_cast<int>(tile_id * blockDim.x + threadIdx.x);
      const int load_pos_a =
          static_cast<int>(row_pos_a * k) + col_pos_a;
      const float reg_a =
          (row_pos_a < static_cast<int>(m) && col_pos_a < static_cast<int>(k))
              ? input_a[load_pos_a]
              : 0.0f;

      const int load_pos_b =
          static_cast<int>(tile_id * blockDim.x + threadIdx.x);
      const float reg_b =
          (load_pos_b < static_cast<int>(k))
              ? g_input_b_constant[load_pos_b]
              : 0.0f;
      rsum += reg_a * reg_b;
    }

    for (size_t lane_offset = 16; lane_offset > 0; lane_offset /= 2) {
      rsum += __shfl_down_sync(0xffffffffu, rsum, lane_offset);
    }

    if (lane_id == 0) {
      atomicAdd(&block_res, rsum);
    }

    __syncthreads();

    if (threadIdx.x == 0 && bx < m) {
      output_c[bx] = block_res;
      block_res = 0.0f;
    }

    __syncthreads();
  }
}

// One-warp-per-output-row GPU kernel.
//
// input_a: device pointer to a row-major matrix with shape (m, k),
//          stored as input_a[row * k + col]
// input_b: device pointer to a vector with shape (k)
// output_c: device pointer to a vector with shape (m)
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
//
// Intended mapping:
// - one warp owns one output row
// - one block owns blockDim.x / 32 output rows
// - lanes in a warp walk columns as col = lane, lane + 32, ...
__global__ void device_mvm_warp_per_row(const float* input_a,
                                        const float* input_b,
                                        float* output_c, size_t m,
                                        size_t k) {
  // Total warps required for output size: one warp per output cell.
  const size_t total_warps = m;

  // For each thread owning one output cell:
  //   grid_stride = blockDim.x * gridDim.x
  //
  // For each block owning one output cell:
  //   grid_stride = gridDim.x
  //
  // For each warp owning one output cell:
  //   warps_per_block = blockDim.x / warpSize
  //   grid_stride = warps_per_block * gridDim.x
  //
  // Example launch config: gridDim.x = 64, blockDim.x = 64
  //   warps_per_block = 64 / 32 = 2
  //   warps_per_grid = 2 * 64 = 128
  const size_t warps_per_block = blockDim.x / warpSize;
  const size_t grid_stride = warps_per_block * gridDim.x;
  const size_t warp_lid = threadIdx.x / warpSize;
  const size_t warp_gid = warps_per_block * blockIdx.x + warp_lid;

  // Example: threadIdx.x 17 and 45 map to lane 17 and lane 13.
  const size_t lane_id = threadIdx.x % warpSize;
  const size_t total_tiles = (k + warpSize - 1) / warpSize;

  // Grid-stride loop at warp level.
  for (size_t wx = warp_gid; wx < total_warps; wx += grid_stride) {
    float rsum = 0.0f;

    // Warp-stride loop: each iteration covers one warp-sized column tile.
    for (size_t tile_id = 0; tile_id < total_tiles; ++tile_id) {
      // Load A register: one warp owns one output cell from m * k by k * 1.
      const size_t row_pos_a = wx;

      // Each warp pulls 32 contiguous A and B elements at a time.
      const size_t col_pos_a = tile_id * warpSize + lane_id;
      const size_t ld_pos_a = row_pos_a * k + col_pos_a;
      const float reg_a =
          (row_pos_a < m && col_pos_a < k) ? input_a[ld_pos_a] : 0.0f;

      // Load B register from the 1D vector.
      const size_t row_pos_b = tile_id * warpSize + lane_id;
      const float reg_b = (row_pos_b < k) ? input_b[row_pos_b] : 0.0f;

      // Lane-local partial sum.
      rsum += reg_a * reg_b;
    }

    // Accumulate lane partial sums into lane 0.
    for (size_t lane_offset = 16; lane_offset > 0; lane_offset /= 2) {
      rsum += __shfl_down_sync(0xffffffffu, rsum, lane_offset);
    }

    // At this point only lane 0 has the full row result.
    if (lane_id == 0 && wx < m) {
      output_c[wx] = rsum;
    }
  }
}

extern "C" void solution(const float* input_a, const float* input_b,
                         float* output_c, size_t m, size_t k) {
  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);
  const size_t shared_bytes =
      2 * static_cast<size_t>(g_launch_config.block_x) * sizeof(float);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      device_mvm_basic<<<grid_shape, block_shape>>>(input_a, input_b, output_c,
                                                    m, k);
      break;
    case KernelVariant::kConstantB:
      device_mvm_constant_b<<<grid_shape, block_shape>>>(input_a, output_c, m,
                                                         k);
      break;
    case KernelVariant::kSharedAB:
      device_mvm_shared_ab<<<grid_shape, block_shape, shared_bytes>>>(
          input_a, input_b, output_c, m, k);
      break;
    case KernelVariant::kWarp:
      device_mvm_warp<<<grid_shape, block_shape>>>(input_a, input_b, output_c,
                                                   m, k);
      break;
    case KernelVariant::kWarpConstB:
      device_mvm_warp_const_b<<<grid_shape, block_shape>>>(input_a, output_c,
                                                           m, k);
      break;
    case KernelVariant::kWarpPerRow:
      device_mvm_warp_per_row<<<grid_shape, block_shape>>>(input_a, input_b,
                                                           output_c, m, k);
      break;
  }

  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_matrix_input(size_t m, size_t k) {
  const size_t total = m * k;
  std::vector<float> input_a(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 37 + m * 11 + k * 7 + 13) % 251) - 125;
    float x = static_cast<float>(raw) / 31.0f;
    if (i % 29 == 0) {
      x = 0.0f;
    } else if (i % 17 == 0) {
      x = -std::fabs(x);
    }
    input_a[i] = x;
  }
  return input_a;
}

static std::vector<float> make_vector_input(size_t k) {
  std::vector<float> input_b(k, 0.0f);
  for (size_t i = 0; i < k; ++i) {
    const int raw = static_cast<int>((i * 19 + k * 5 + 23) % 127) - 63;
    float x = static_cast<float>(raw) / 29.0f;
    if (i % 31 == 0) {
      x = 0.0f;
    } else if (i % 11 == 0) {
      x = -std::fabs(x);
    }
    input_b[i] = x;
  }
  return input_b;
}

static bool verify_close(const std::vector<float>& got,
                         const std::vector<float>& expected, float atol,
                         float rtol, const char* label, bool verbose) {
  if (got.size() != expected.size()) {
    if (verbose) {
      std::cerr << "verify(" << label << "): size mismatch got=" << got.size()
                << " expected=" << expected.size() << '\n';
    }
    return false;
  }

  float max_abs = 0.0f;
  size_t max_i = 0;
  bool ok = true;
  size_t first_bad = 0;

  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) {
      if (ok) {
        ok = false;
        first_bad = i;
      }
      continue;
    }
    const float diff = std::fabs(got[i] - expected[i]);
    if (diff > max_abs) {
      max_abs = diff;
      max_i = i;
    }
    const float tol = atol + rtol * std::fabs(expected[i]);
    if (diff > tol && ok) {
      ok = false;
      first_bad = i;
    }
  }

  if (!ok) {
    if (verbose) {
      std::cerr << "verify(" << label << "): FAIL at i=" << first_bad
                << " got=" << got[first_bad]
                << " expected=" << expected[first_bad]
                << " max_abs=" << max_abs << " max_i=" << max_i << '\n';
    }
    return false;
  }

  if (verbose) {
    std::cerr << "verify(" << label << "): PASS max_abs=" << max_abs
              << " max_i=" << max_i << '\n';
  }
  return true;
}

struct TestResult {
  const char* group = "";
  const char* name = "";
  const char* kernel = "";
  size_t m = 0;
  size_t k = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group" << std::setw(16) << "name"
            << std::setw(14) << "kernel" << std::setw(10) << "m"
            << std::setw(10) << "k" << std::setw(8) << "block_x"
            << std::setw(8) << "grid_x" << std::setw(6) << "cpu"
            << std::setw(6) << "gpu" << std::setw(12) << "total_ms"
            << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(110, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& r : results) {
    std::cout << std::left << std::setw(8) << r.group << std::setw(16)
              << r.name << std::setw(14) << r.kernel << std::setw(10) << r.m
              << std::setw(10) << r.k << std::setw(8) << r.block_x
              << std::setw(8) << r.grid_x << std::setw(6) << r.cpu
              << std::setw(6) << r.gpu << std::setw(12) << r.total_ms
              << std::setw(12) << r.kernel_ms << '\n';
  }
}

static void print_scale_heatmaps(const std::vector<TestResult>& results) {
  std::vector<std::string> names;
  std::vector<std::string> kernels;
  std::vector<int> block_sizes;
  std::vector<int> grid_sizes;

  for (const auto& r : results) {
    if (r.group != std::string("scale") || r.gpu == "FAIL") {
      continue;
    }
    if (std::find(names.begin(), names.end(), r.name) == names.end()) {
      names.push_back(r.name);
    }
    if (std::find(kernels.begin(), kernels.end(), r.kernel) == kernels.end()) {
      kernels.push_back(r.kernel);
    }
    if (std::find(block_sizes.begin(), block_sizes.end(), r.block_x) ==
        block_sizes.end()) {
      block_sizes.push_back(r.block_x);
    }
    if (std::find(grid_sizes.begin(), grid_sizes.end(), r.grid_x) ==
        grid_sizes.end()) {
      grid_sizes.push_back(r.grid_x);
    }
  }

  if (names.empty()) {
    return;
  }

  std::sort(block_sizes.begin(), block_sizes.end());
  std::sort(grid_sizes.begin(), grid_sizes.end());

  std::cout << "\nScaling Heatmaps (kernel_ms, lower is better)\n";
  std::cout << std::string(60, '=') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& name : names) {
    for (const auto& kernel : kernels) {
      float best_ms = -1.0f;
      int best_block = 0;
      int best_grid = 0;

      for (const auto& r : results) {
        if (r.group == std::string("scale") && r.name == name &&
            r.kernel == kernel && r.gpu != "FAIL") {
          if (best_ms < 0.0f || r.kernel_ms < best_ms) {
            best_ms = r.kernel_ms;
            best_block = r.block_x;
            best_grid = r.grid_x;
          }
        }
      }

      std::cout << '\n' << name << " / " << kernel << "  best=(" << best_block
                << ", " << best_grid << ") -> " << best_ms << " ms\n";
      std::cout << std::left << std::setw(10) << "block\\grid";
      for (int grid_x : grid_sizes) {
        std::cout << std::setw(10) << grid_x;
      }
      std::cout << '\n';

      for (int block_x : block_sizes) {
        std::cout << std::left << std::setw(10) << block_x;
        for (int grid_x : grid_sizes) {
          bool found = false;
          for (const auto& r : results) {
            if (r.group == std::string("scale") && r.name == name &&
                r.kernel == kernel && r.block_x == block_x &&
                r.grid_x == grid_x && r.gpu != "FAIL") {
              std::cout << std::setw(10) << r.kernel_ms;
              found = true;
              break;
            }
          }
          if (!found) {
            std::cout << std::setw(10) << "-";
          }
        }
        std::cout << '\n';
      }
    }
  }
}

static void collect_kernel_timing_samples(const float* d_input_a,
                                          const float* d_input_b,
                                          float* d_output_c, size_t bytes_c,
                                          size_t m, size_t k,
                                          cudaEvent_t kernel_start,
                                          cudaEvent_t kernel_stop,
                                          std::vector<float>& samples) {
  for (int i = 0; i < kTimingWarmupIterations; ++i) {
    CUDA_CHECK(cudaMemset(d_output_c, 0, bytes_c));
    solution(d_input_a, d_input_b, d_output_c, m, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  samples.clear();
  samples.reserve(g_timing_repeats);
  for (int i = 0; i < g_timing_repeats; ++i) {
    CUDA_CHECK(cudaMemset(d_output_c, 0, bytes_c));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    solution(d_input_a, d_input_b, d_output_c, m, k);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    samples.push_back(kernel_ms);
  }
}

static void run_solution_host_global_b(const std::vector<float>& input_a,
                                       const std::vector<float>& input_b,
                                       std::vector<float>& output_c, size_t m,
                                       size_t k) {
  cudaEvent_t total_start = nullptr;
  cudaEvent_t total_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  const size_t bytes_a = m * k * sizeof(float);
  const size_t bytes_b = k * sizeof(float);
  const size_t bytes_c = m * sizeof(float);

  float* d_input_a = nullptr;
  float* d_input_b = nullptr;
  float* d_output_c = nullptr;

  CUDA_CHECK(cudaEventRecord(total_start));
  CUDA_CHECK(cudaMalloc(&d_input_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_input_b, bytes_b));
  CUDA_CHECK(cudaMalloc(&d_output_c, bytes_c));
  CUDA_CHECK(cudaMemcpy(d_input_a, input_a.data(), bytes_a,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input_b, input_b.data(), bytes_b,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output_c, 0, bytes_c));

  std::vector<float> kernel_samples;
  collect_kernel_timing_samples(d_input_a, d_input_b, d_output_c, bytes_c, m,
                                k, kernel_start, kernel_stop,
                                kernel_samples);

  CUDA_CHECK(cudaMemcpy(output_c.data(), d_output_c, bytes_c,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));
  g_last_timing.total_ms = total_ms;
  g_last_timing.kernel_ms = select_timing_sample(kernel_samples);

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_input_a));
  CUDA_CHECK(cudaFree(d_input_b));
  CUDA_CHECK(cudaFree(d_output_c));
}

static void run_solution_host_constant_b(const std::vector<float>& input_a,
                                         const std::vector<float>& input_b,
                                         std::vector<float>& output_c,
                                         size_t m, size_t k) {
  if (k > kConstantInputBElements) {
    std::cerr << "constant-memory kernels support k <= "
              << kConstantInputBElements << ", got " << k << '\n';
    std::exit(EXIT_FAILURE);
  }

  cudaEvent_t total_start = nullptr;
  cudaEvent_t total_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  const size_t bytes_a = m * k * sizeof(float);
  const size_t bytes_b = k * sizeof(float);
  const size_t bytes_c = m * sizeof(float);

  float* d_input_a = nullptr;
  float* d_output_c = nullptr;

  CUDA_CHECK(cudaEventRecord(total_start));
  CUDA_CHECK(cudaMalloc(&d_input_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_output_c, bytes_c));
  CUDA_CHECK(cudaMemcpy(d_input_a, input_a.data(), bytes_a,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(g_input_b_constant, input_b.data(), bytes_b));
  CUDA_CHECK(cudaMemset(d_output_c, 0, bytes_c));

  std::vector<float> kernel_samples;
  collect_kernel_timing_samples(d_input_a, nullptr, d_output_c, bytes_c, m, k,
                                kernel_start, kernel_stop, kernel_samples);

  CUDA_CHECK(cudaMemcpy(output_c.data(), d_output_c, bytes_c,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));
  g_last_timing.total_ms = total_ms;
  g_last_timing.kernel_ms = select_timing_sample(kernel_samples);

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_input_a));
  CUDA_CHECK(cudaFree(d_output_c));
}

static void run_solution_host(const std::vector<float>& input_a,
                              const std::vector<float>& input_b,
                              std::vector<float>& output_c, size_t m,
                              size_t k) {
  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
    case KernelVariant::kSharedAB:
    case KernelVariant::kWarp:
    case KernelVariant::kWarpPerRow:
      run_solution_host_global_b(input_a, input_b, output_c, m, k);
      break;
    case KernelVariant::kConstantB:
    case KernelVariant::kWarpConstB:
      run_solution_host_constant_b(input_a, input_b, output_c, m, k);
      break;
  }
}

static int run_profile() {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const size_t m = 32768;
  const size_t k = 4096;
  const auto input_a = make_matrix_input(m, k);
  const auto input_b = make_vector_input(k);
  const size_t bytes_a = m * k * sizeof(float);
  const size_t bytes_b = k * sizeof(float);
  const size_t bytes_c = m * sizeof(float);

  if (uses_constant_b(g_kernel_variant) && k > kConstantInputBElements) {
    std::cerr << "constant-memory kernels support k <= "
              << kConstantInputBElements << ", got " << k << '\n';
    return 1;
  }

  float* d_input_a = nullptr;
  float* d_input_b = nullptr;
  float* d_output_c = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_output_c, bytes_c));
  CUDA_CHECK(cudaMemcpy(d_input_a, input_a.data(), bytes_a,
                        cudaMemcpyHostToDevice));
  if (uses_constant_b(g_kernel_variant)) {
    CUDA_CHECK(cudaMemcpyToSymbol(g_input_b_constant, input_b.data(),
                                  bytes_b));
  } else {
    CUDA_CHECK(cudaMalloc(&d_input_b, bytes_b));
    CUDA_CHECK(cudaMemcpy(d_input_b, input_b.data(), bytes_b,
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemset(d_output_c, 0, bytes_c));

  for (int i = 0; i < kProfileWarmupIterations; ++i) {
    solution(d_input_a, d_input_b, d_output_c, m, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));
  CUDA_CHECK(cudaEventRecord(kernel_start));
  for (int i = 0; i < kProfileIterations; ++i) {
    solution(d_input_a, d_input_b, d_output_c, m, k);
  }
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
  const float avg_kernel_ms = kernel_ms / kProfileIterations;
  const size_t device_bytes =
      bytes_a + bytes_c +
      (uses_constant_b(g_kernel_variant) ? 0 : bytes_b);
  const double device_mib =
      static_cast<double>(device_bytes) / (1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(3)
            << "profile scope=kernel-only verify=off"
            << " kernel=" << current_kernel_name()
            << " b_mode="
            << (uses_constant_b(g_kernel_variant)
                    ? "constant-preloaded-fixed-b"
                    : "device-pointer")
            << " m=" << m
            << " k=" << k << " block_x=" << g_launch_config.block_x
            << " grid_x=" << g_launch_config.grid_x
            << " warmup=" << kProfileWarmupIterations
            << " repeats=" << kProfileIterations
            << " device_mib=" << device_mib
            << " avg_kernel_ms=" << avg_kernel_ms << '\n';

  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_input_a));
  if (d_input_b != nullptr) {
    CUDA_CHECK(cudaFree(d_input_b));
  }
  CUDA_CHECK(cudaFree(d_output_c));
  return 0;
}

static int run_tests(bool skip_cpu_verify) {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const LaunchConfig default_launch = g_launch_config;
  const KernelVariant kernel_variants[] = {
      KernelVariant::kBasic,
      KernelVariant::kConstantB,
      KernelVariant::kSharedAB,
      KernelVariant::kWarp,
      KernelVariant::kWarpConstB,
      KernelVariant::kWarpPerRow,
  };

  const std::vector<TestCase> tests = {
      {"small_1",
       2,
       3,
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
       {1.0f, 0.0f, -1.0f},
       {-2.0f, -2.0f}},
      {"small_2",
       3,
       1,
       {-2.0f, 0.0f, 3.5f},
       {2.0f},
       {-4.0f, 0.0f, 7.0f}},
      {"small_3",
       1,
       4,
       {0.5f, -1.0f, 2.0f, -4.0f},
       {2.0f, -3.0f, 0.25f, -0.5f},
       {6.5f}},
      {"small_4",
       3,
       4,
       {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 1.5f, 0.5f, 0.0f, 1.0f,
        0.0f, 2.0f},
       {0.25f, -1.0f, 2.0f, 0.5f},
       {6.25f, 5.0f, 0.0f}},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } medium_tests[] = {
      {"medium_1", 64, 64},
      {"medium_2", 255, 257},
      {"medium_3", 513, 1025},
      {"medium_4", 1024, 1024},
      {"medium_tail", 257, 258},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } large_verify_tests[] = {
      {"large_1", 1023, 2049},
      {"large_2", 1537, 2049},
  };

  const struct {
    const char* name;
    const char* scale_name;
    size_t m;
    size_t k;
  } tensara_tests[] = {
      {"tensara_1", "scale_tensara_1", 4096, 4096},
      {"tensara_2", "scale_tensara_2", 6144, 4096},
      {"tensara_3", "scale_tensara_3", 7168, 4096},
      {"tensara_4", "scale_tensara_4", 8192, 4096},
      {"tensara_5", "scale_tensara_5", 9216, 4096},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } shape_tests[] = {
      {"shape_tall", 8192, 1024},
      {"shape_wide", 1024, 8192},
      {"shape_odd", 4097, 2049},
      {"shape_rect", 2049, 4097},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } scale_tests[] = {
      {"scale_sq", 4096, 4096},
      {"scale_tall", 8192, 2048},
      {"scale_wide", 2048, 8192},
  };

  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};

  bool all_ok = true;
  std::vector<TestResult> results;

  auto run_sized = [&](const char* group, const char* name, size_t m,
                       size_t k) {
    g_launch_config = default_launch;

    const auto input_a = make_matrix_input(m, k);
    const auto input_b = make_vector_input(k);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref.assign(m, 0.0f);
      cpu_matrix_vector(input_a, input_b, ref, m, k);
      cpu_status = "REF";
    }

    std::vector<float> gpu_out(m, 0.0f);
    run_solution_host(input_a, input_b, gpu_out, m, k);

    TestResult res;
    res.group = group;
    res.name = name;
    res.kernel = current_kernel_name();
    res.m = m;
    res.k = k;
    res.block_x = g_launch_config.block_x;
    res.grid_x = g_launch_config.grid_x;
    res.cpu = cpu_status;

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      const bool gpu_ok =
          verify_close(gpu_out, ref, 1e-3f, 1e-3f, name, false);
      all_ok &= gpu_ok;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
    } else {
      res.gpu = "SKIP";
    }

    res.total_ms = g_last_timing.total_ms;
    res.kernel_ms = g_last_timing.kernel_ms;
    results.push_back(res);
  };

  auto run_scaling = [&](const char* name, size_t m, size_t k) {
    const auto input_a = make_matrix_input(m, k);
    const auto input_b = make_vector_input(k);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref.assign(m, 0.0f);
      cpu_matrix_vector(input_a, input_b, ref, m, k);
      cpu_status = "REF";
    }

    for (int block_x : scale_block_sizes) {
      for (int grid_x : scale_grid_sizes) {
        g_launch_config = {block_x, grid_x};

        std::vector<float> gpu_out(m, 0.0f);
        run_solution_host(input_a, input_b, gpu_out, m, k);

        TestResult res;
        res.group = "scale";
        res.name = name;
        res.kernel = current_kernel_name();
        res.m = m;
        res.k = k;
        res.block_x = g_launch_config.block_x;
        res.grid_x = g_launch_config.grid_x;
        res.cpu = cpu_status;

        if (!skip_cpu_verify && kCpuReferenceImplemented) {
          const bool gpu_ok =
              verify_close(gpu_out, ref, 1e-3f, 1e-3f, name, false);
          all_ok &= gpu_ok;
          res.gpu = gpu_ok ? "PASS" : "FAIL";
        } else {
          res.gpu = "SKIP";
        }

        res.total_ms = g_last_timing.total_ms;
        res.kernel_ms = g_last_timing.kernel_ms;
        results.push_back(res);
      }
    }
  };

  for (KernelVariant kernel_variant : kernel_variants) {
    if (!kernel_enabled(kernel_variant)) {
      continue;
    }
    g_kernel_variant = kernel_variant;

    for (const auto& tc : tests) {
      g_launch_config = default_launch;

      std::string cpu_status = "SKIP";
      if (!skip_cpu_verify && kCpuReferenceImplemented) {
        std::vector<float> ref(tc.m, 0.0f);
        cpu_matrix_vector(tc.input_a, tc.input_b, ref, tc.m, tc.k);
        const bool cpu_ok =
            verify_close(ref, tc.expected, 1e-5f, 1e-5f, tc.name, false);
        cpu_status = cpu_ok ? "PASS" : "FAIL";
        all_ok &= cpu_ok;
      }

      std::vector<float> gpu_out(tc.m, 0.0f);
      run_solution_host(tc.input_a, tc.input_b, gpu_out, tc.m, tc.k);

      const bool gpu_ok =
          verify_close(gpu_out, tc.expected, 1e-4f, 1e-4f, tc.name, false);
      all_ok &= gpu_ok;

      TestResult res;
      res.group = "small";
      res.name = tc.name;
      res.kernel = current_kernel_name();
      res.m = tc.m;
      res.k = tc.k;
      res.block_x = g_launch_config.block_x;
      res.grid_x = g_launch_config.grid_x;
      res.cpu = cpu_status;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
      res.total_ms = g_last_timing.total_ms;
      res.kernel_ms = g_last_timing.kernel_ms;
      results.push_back(res);
    }

    for (const auto& mt : medium_tests) {
      run_sized("medium", mt.name, mt.m, mt.k);
    }

    for (const auto& lt : large_verify_tests) {
      run_sized("large", lt.name, lt.m, lt.k);
    }

    if (!skip_cpu_verify) {
      run_scaling("scale_verify", 257, 383);
    }

    if (skip_cpu_verify) {
      for (const auto& tt : tensara_tests) {
        run_sized("tensara", tt.name, tt.m, tt.k);
      }
      for (const auto& tt : tensara_tests) {
        run_scaling(tt.scale_name, tt.m, tt.k);
      }
      for (const auto& st : shape_tests) {
        run_sized("shape", st.name, st.m, st.k);
      }
      for (const auto& sc : scale_tests) {
        run_scaling(sc.name, sc.m, sc.k);
      }
    }
  }

  g_kernel_variant = KernelVariant::kBasic;
  g_launch_config = default_launch;
  std::cout << "Timing samples: mode=" << timing_mode_name()
            << " repeats=" << g_timing_repeats
            << " warmup=" << kTimingWarmupIterations
            << " metric=kernel_ms\n\n";
  print_results_table(results);
  print_scale_heatmaps(results);
  return all_ok ? 0 : 1;
}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  bool skip_cpu_verify = false;
  bool profile_mode = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--skip-cpu") {
      skip_cpu_verify = true;
    } else if (std::string(argv[i]) == "--profile") {
      profile_mode = true;
    } else if (std::string(argv[i]).rfind("--kernel=", 0) == 0) {
      if (!parse_kernel_arg(argv[i])) {
        return 1;
      }
    } else if (std::string(argv[i]).rfind("--timing=", 0) == 0) {
      if (!parse_timing_arg(argv[i])) {
        return 1;
      }
    } else if (std::string(argv[i]).rfind("--timing-repeats=", 0) == 0) {
      if (!parse_timing_repeats_arg(argv[i])) {
        return 1;
      }
    } else {
      std::cerr << "Unknown argument: " << argv[i]
                << " (supported: --skip-cpu, --profile, --kernel=..., "
                << "--timing=..., --timing-repeats=...)\n";
      return 1;
    }
  }

  if (profile_mode) {
    if (!g_kernel_arg_set) {
      std::cerr << "--profile requires an explicit --kernel=... value\n";
      return 1;
    }
    return run_profile();
  }
  return run_tests(skip_cpu_verify);
}
