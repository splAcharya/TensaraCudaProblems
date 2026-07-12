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

/*
 * Problem 9: RMS Normalization
 * Source: https://tensara.org/problems/rms-norm
 *
 * Given an input tensor X with shape B x N, compute output tensor Y with the
 * same shape by normalizing each batch row by its root mean square over the
 * feature dimension.
 *
 * For each row b:
 *
 *   mean_sq[b] = (1 / N) * sum over j of X[b][j]^2
 *   rms[b]     = sqrt(mean_sq[b] + 1e-5)
 *   Y[b][j]    = X[b][j] / rms[b]
 *
 * Input/output shape rules:
 * - X is a row-major float32 matrix with shape B x N.
 * - Y is a row-major float32 matrix with shape B x N.
 * - The published problem definition passes extra params as [B, N].
 * - The local solution signature therefore uses B as rows and N as features.
 *
 * Important notes:
 * - RMS is calculated independently for each row over dimension 1.
 * - Use epsilon = 1e-5 for numerical stability.
 * - The problem definition verifies with rtol=2e-4 and atol=1e-4.
 * - Embedded problem data confirmed the parameter order: X, Y, B, N.
 *
 * Published Tensara sizes:
 * - shape=(1024, 1024)
 * - shape=(1024, 4096)
 * - shape=(2048, 8192)
 * - shape=(512, 16384)
 */

// Tensara-style signature:
// - X and Y are device pointers
// - B is the batch size / row count
// - N is the feature count / column count
// - X/Y are row-major matrices with shape (B, N)
extern "C" void solution(const float* X, float* Y, size_t B, size_t N);

static constexpr bool kCpuReferenceImplemented = true;
static constexpr bool kGpuKernelImplemented = true;
static constexpr int kDefaultTimingRepeats = 5;
static constexpr int kTimingWarmupIterations = 1;
static constexpr int kProfileWarmupIterations = 5;
static constexpr int kProfileIterations = 50;

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
  kFloat4,
  kSharedMem,
  kWarp,
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
    case KernelVariant::kFloat4:
      return "float4";
    case KernelVariant::kSharedMem:
      return "shared_mem";
    case KernelVariant::kWarp:
      return "warp";
  }
  return "unknown";
}

static bool kernel_enabled(KernelVariant variant) {
  return !g_kernel_arg_set || g_kernel_variant == variant;
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
  if (value == "float4") {
    g_kernel_variant = KernelVariant::kFloat4;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "shared_mem") {
    g_kernel_variant = KernelVariant::kSharedMem;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "warp") {
    g_kernel_variant = KernelVariant::kWarp;
    g_kernel_arg_set = true;
    return true;
  }

  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic, --kernel=float4, "
            << "--kernel=shared_mem, or --kernel=warp)\n";
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
  size_t rows = 0;
  size_t cols = 0;
  std::vector<float> input;
  std::vector<float> expected;
};

struct SizedCase {
  const char* name = "";
  size_t rows = 0;
  size_t cols = 0;
};

struct TestResult {
  std::string group;
  std::string name;
  std::string kernel;
  size_t rows = 0;
  size_t cols = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

// CPU reference implementation.
// X: row-major matrix flattened with shape (rows * cols)
// Y: row-major matrix flattened with shape (rows * cols)
// rows: batch size B
// cols: feature count N
static void cpu_rms_norm_reference(const std::vector<float>& X,
                                   std::vector<float>& Y, size_t rows,
                                   size_t cols)
{
  for (int b = 0; b < rows; ++b)
  {
    float sum_sq = 0.0f;

    for (int j = 0; j < cols; ++j)
    {
      const float temp_x =  X[b * cols + j];
      sum_sq += (temp_x * temp_x);
    }

    float mean_sq = sum_sq / cols;
    float rms = std::sqrt(mean_sq + 1e-5f);

    for (int j = 0; j < cols; ++j)
      Y[b *cols + j] = X[b * cols + j] / rms;
  }
}

// Basic GPU kernel implementation.
//
// X: device pointer to row-major matrix with shape (B, N)
// Y: device pointer to row-major matrix with shape (B, N)
// B: batch size / row count
// N: feature count / column count
__global__ void rms_norm_basic_kernel(
  const float* X,
  float* Y,
  size_t B,
  size_t N)
{
  size_t gix = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);
  size_t total = B * N;

  //each thread
  for (size_t gx = gix; gx < total; gx += grid_stride)
  {
    //get row/col pos
    size_t row = gx / N;
    size_t col = gx % N;
    float sum_sq = 0.0f;

    for (int j = 0; j < N; ++j)
    {
      const float temp_x = X[row * N + j];
      sum_sq += (temp_x * temp_x);
    }

    float mean_sq = sum_sq / N;
    float rms = std::sqrt(mean_sq + 1e-5f);
    Y[row * N + col] = X[row * N + col] / rms;
  }
}

// Float4 GPU kernel implementation.
//
// X: device pointer to row-major matrix with shape (B, N)
// Y: device pointer to row-major matrix with shape (B, N)
// B: batch size / row count
// N: feature count / column count
// Each thread computes one output and uses scalar prefix/tail elements
// around aligned float4 loads when accumulating the row RMS.
__global__ void rms_norm_float4_kernel(const float* X, float* Y,
                                       size_t B, size_t N)                                        
{
  size_t total = B * N;
  size_t gix   = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t total_threads = (blockDim.x * gridDim.x); //i.e grid stride

  for (size_t gx = gix; gx < total; gx += total_threads)
  {
    //determine row idx
    size_t row_pos = gx / N;
    size_t col_pos = gx % N;

    //determine scalar prefix
    size_t pos_in_f4     = (row_pos * N) % 4;  
    size_t prefix_count  = (4 - pos_in_f4) % 4;

    //determine vector chunks
    size_t vector_remain = N - prefix_count;
    size_t vector_count  = vector_remain / 4;

    //determine scalar tail
    size_t tail_start = prefix_count + (vector_count * 4);

    //process scalar prefix
    float sum_sq = 0.0f;
    for (size_t col = 0; col < prefix_count; ++col)
    {
      const float temp_x = X[row_pos * N + col];
      sum_sq += (temp_x * temp_x);
    }

    //process float 4 chunks
    const float4 *X4 = reinterpret_cast<const float4 *>(&(X[row_pos * N + prefix_count]));
    for (size_t col = 0; col < vector_count; ++col)
    {
      const float4 temp_x = X4[col];
      sum_sq += (temp_x.x * temp_x.x);
      sum_sq += (temp_x.y * temp_x.y);
      sum_sq += (temp_x.z * temp_x.z);
      sum_sq += (temp_x.w * temp_x.w);
    }

    //process scalar tail
    for (size_t col = tail_start; col < N; ++col)
    {
      const float temp_x  = X[row_pos * N + col];
      sum_sq             += (temp_x * temp_x); 
    }

    //compute rms norm
    float mean_sq    = sum_sq / N;
    float rms        = std::sqrt(mean_sq + 1e-5f);
    Y[gx]            = X[gx] / rms;
  }
}

// Shared-memory GPU kernel implementation.
//
// X: device pointer to row-major matrix with shape (B, N)
// Y: device pointer to row-major matrix with shape (B, N)
// B: batch size / row count
// N: feature count / column count
__global__ void rms_norm_shared_mem_kernel(const float* X, float* Y,
                                           size_t B, size_t N) {

  extern __shared__ float smem_tile[];
  size_t total = B * N;

  //grid stride loop at block level
  for (size_t bx = blockIdx.x; bx < B; bx += gridDim.x)
  {
    // Shared memory size matches the block size, so each thread clears one slot.
    smem_tile[threadIdx.x] = 0.0f;

    __syncthreads();

    //reduction at block level-->each block gets an entire row;
    for (size_t lx = threadIdx.x; lx < N; lx += blockDim.x)
    {
      size_t flat_idx = bx * N + lx;
      const float temp_x = (flat_idx < total) ? X[flat_idx] : 0.0f;
      const float temp_x_sq = temp_x * temp_x;
      smem_tile[threadIdx.x] += temp_x_sq;
    }

    __syncthreads(); //wait for entire block to write data into smem

    //shared memory tiles have some portion of the data, now
    //reduce that to a single sum
    for (size_t offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
      //only half of the block participates each iteration one half goes away
      //at the end only smem[0] will have the final sum
      if (threadIdx.x < offset)
        smem_tile[threadIdx.x] += smem_tile[threadIdx.x + offset];

      __syncthreads();
    }

    if (threadIdx.x == 0)
    {
      float sum_sq = smem_tile[0];
      float mean_sq = sum_sq / N;
      float inv_rms = 1.00f / std::sqrt(mean_sq + 1e-5f);
      smem_tile[0] = inv_rms;
    }
    __syncthreads();

    // smem[0] now holds the inverse rms.
    for (size_t lx = threadIdx.x; lx < N; lx += blockDim.x)
    {
      size_t flat_idx = bx * N + lx;
      Y[flat_idx] = X[flat_idx] * smem_tile[0];
    }
  }
}

__global__ void rms_norm_warp_kernel(const float* X, float* Y,
                                     size_t B, size_t N) {

  const size_t total_warps = B;
  const size_t warps_per_block = blockDim.x / warpSize;
  const size_t warps_per_grid = gridDim.x * warps_per_block;
  const size_t warp_lid = threadIdx.x / warpSize;
  const size_t lane_id = threadIdx.x % warpSize;
  const size_t warp_gid = (blockIdx.x * warps_per_block) + warp_lid;
  
  //grid stride loop @ warp level
  for (size_t wx = warp_gid; wx < total_warps; wx += warps_per_grid)
  {
    //each warp gets an entire output row,

    //load data locally in warps registers i.e do warp stride loop @ lane level
    float lane_sum_sq = 0.0f;
    for (int lx = lane_id; lx < N; lx += warpSize)
    {
      size_t flat_idx = (wx * N) + lx;
      const float temp_x = X[flat_idx];
      lane_sum_sq += (temp_x * temp_x);
    }

    //warp shuffle reduction
    const unsigned mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      lane_sum_sq += __shfl_down_sync(mask, lane_sum_sq, offset);
    
    //value now should be with lane 0, shre wit witho toehrs
    float send_inv_rms = 0.0f;
    if (lane_id == 0)
    {
      float sum_sq = lane_sum_sq;
      float mean_sq = sum_sq / N;
      float inv_rms = 1.00f / std::sqrt(mean_sq + 1e-5f);
      send_inv_rms = inv_rms;
    }
    
    float recv_inv_rms  = __shfl_sync(mask, send_inv_rms, 0);

    //warp stride loop @ lane level to compute outputs
    for (size_t lx = lane_id; lx < N; lx += warpSize)
    {
      size_t flat_idx = (wx * N) + lx;
      Y[flat_idx] = X[flat_idx] * recv_inv_rms;
    }
  }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
  if (!kGpuKernelImplemented) {
    return;
  }
  if (B == 0 || N == 0) {
    return;
  }

  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      rms_norm_basic_kernel<<<grid_shape, block_shape>>>(X, Y, B, N);
      break;
    case KernelVariant::kFloat4:
      rms_norm_float4_kernel<<<grid_shape, block_shape>>>(X, Y, B, N);
      break;
    case KernelVariant::kSharedMem: {
      size_t smem_bytes = block_shape.x * sizeof(float);
      rms_norm_shared_mem_kernel<<<grid_shape, block_shape, smem_bytes>>>(
          X, Y, B, N);
      break;
    }
    case KernelVariant::kWarp:
      rms_norm_warp_kernel<<<grid_shape, block_shape>>>(X, Y, B, N);
      break;
  }

  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_rms_norm_input(size_t rows, size_t cols) {
  const size_t total = rows * cols;
  std::vector<float> input(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 37 + rows * 11 + cols * 17 + 19) % 257) - 128;
    input[i] = static_cast<float>(raw) / 32.0f;
  }
  return input;
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
    std::cout << "verify(" << label << "): PASS max_abs=" << max_abs
              << " max_i=" << max_i << '\n';
  }
  return true;
}

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group" << std::setw(18)
            << "name" << std::setw(12) << "kernel" << std::setw(10)
            << "rows" << std::setw(10) << "cols" << std::setw(8)
            << "block_x" << std::setw(8) << "grid_x" << std::setw(6)
            << "cpu" << std::setw(6) << "gpu" << std::setw(12)
            << "total_ms" << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(110, '-') << '\n';
  for (const auto& r : results) {
    std::cout << std::left << std::setw(8) << r.group << std::setw(18)
              << r.name << std::setw(12) << r.kernel << std::setw(10)
              << r.rows << std::setw(10) << r.cols << std::setw(8)
              << r.block_x << std::setw(8) << r.grid_x << std::setw(6)
              << r.cpu << std::setw(6) << r.gpu << std::setw(12)
              << std::fixed << std::setprecision(3) << r.total_ms
              << std::setw(12) << r.kernel_ms << '\n';
  }
}

static void print_scale_heatmaps(const std::vector<TestResult>& results) {
  std::vector<std::string> names;
  std::vector<std::string> kernels;
  std::vector<int> block_sizes;
  std::vector<int> grid_sizes;

  for (const auto& res : results) {
    if (res.group != "scale") {
      continue;
    }
    if (std::find(names.begin(), names.end(), res.name) == names.end()) {
      names.push_back(res.name);
    }
    if (std::find(kernels.begin(), kernels.end(), res.kernel) ==
        kernels.end()) {
      kernels.push_back(res.kernel);
    }
    if (std::find(block_sizes.begin(), block_sizes.end(), res.block_x) ==
        block_sizes.end()) {
      block_sizes.push_back(res.block_x);
    }
    if (std::find(grid_sizes.begin(), grid_sizes.end(), res.grid_x) ==
        grid_sizes.end()) {
      grid_sizes.push_back(res.grid_x);
    }
  }

  if (names.empty()) {
    return;
  }

  std::sort(names.begin(), names.end());
  std::sort(kernels.begin(), kernels.end());
  std::sort(block_sizes.begin(), block_sizes.end());
  std::sort(grid_sizes.begin(), grid_sizes.end());

  std::cout << "\nScale heatmaps: kernel_ms\n";

  for (const auto& name : names) {
    for (const auto& kernel : kernels) {
      float best_ms = 0.0f;
      int best_block = 0;
      int best_grid = 0;
      bool have_best = false;

      for (const auto& res : results) {
        if (res.group == "scale" && res.name == name &&
            res.kernel == kernel && res.gpu != "FAIL" &&
            (!have_best || res.kernel_ms < best_ms)) {
          best_ms = res.kernel_ms;
          best_block = res.block_x;
          best_grid = res.grid_x;
          have_best = true;
        }
      }

      if (!have_best) {
        continue;
      }

      std::cout << '\n' << name << " / " << kernel << " best=("
                << best_block << ", " << best_grid << ") -> " << best_ms
                << " ms\n";
      std::cout << std::left << std::setw(10) << "block\\grid";
      for (size_t i = 0; i < grid_sizes.size(); ++i) {
        if (i + 1 == grid_sizes.size()) {
          std::cout << grid_sizes[i];
        } else {
          std::cout << std::setw(10) << grid_sizes[i];
        }
      }
      std::cout << '\n';

      for (int block_x : block_sizes) {
        std::cout << std::left << std::setw(10) << block_x;
        for (size_t i = 0; i < grid_sizes.size(); ++i) {
          const int grid_x = grid_sizes[i];
          const bool last_grid = i + 1 == grid_sizes.size();
          bool found = false;
          for (const auto& res : results) {
            if (res.group == "scale" && res.name == name &&
                res.kernel == kernel && res.block_x == block_x &&
                res.grid_x == grid_x && res.gpu != "FAIL") {
              if (last_grid) {
                std::cout << res.kernel_ms;
              } else {
                std::cout << std::setw(10) << res.kernel_ms;
              }
              found = true;
              break;
            }
          }
          if (!found) {
            if (last_grid) {
              std::cout << "-";
            } else {
              std::cout << std::setw(10) << "-";
            }
          }
        }
        std::cout << '\n';
      }
    }
  }
}

static void collect_kernel_timing_samples(const float* d_X, float* d_Y,
                                          size_t bytes, size_t rows,
                                          size_t cols,
                                          cudaEvent_t kernel_start,
                                          cudaEvent_t kernel_stop,
                                          std::vector<float>& samples) {
  for (int i = 0; i < kTimingWarmupIterations; ++i) {
    CUDA_CHECK(cudaMemset(d_Y, 0, bytes));
    solution(d_X, d_Y, rows, cols);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  samples.clear();
  samples.reserve(static_cast<size_t>(g_timing_repeats));
  for (int i = 0; i < g_timing_repeats; ++i) {
    CUDA_CHECK(cudaMemset(d_Y, 0, bytes));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    solution(d_X, d_Y, rows, cols);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    samples.push_back(kernel_ms);
  }
}

static std::vector<float> run_gpu_case(const std::vector<float>& input,
                                       size_t rows, size_t cols) {
  const size_t total = rows * cols;
  const size_t bytes = total * sizeof(float);
  std::vector<float> output(total, 0.0f);
  float* d_X = nullptr;
  float* d_Y = nullptr;
  cudaEvent_t total_start = nullptr;
  cudaEvent_t total_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;

  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));
  CUDA_CHECK(cudaEventRecord(total_start));

  CUDA_CHECK(cudaMalloc(&d_X, bytes));
  CUDA_CHECK(cudaMalloc(&d_Y, bytes));
  CUDA_CHECK(cudaMemcpy(d_X, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_Y, 0, bytes));

  std::vector<float> kernel_samples;
  collect_kernel_timing_samples(d_X, d_Y, bytes, rows, cols, kernel_start,
                                kernel_stop, kernel_samples);
  CUDA_CHECK(cudaMemcpy(output.data(), d_Y, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));
  CUDA_CHECK(cudaEventElapsedTime(&g_last_timing.total_ms, total_start,
                                  total_stop));
  g_last_timing.kernel_ms = select_timing_sample(kernel_samples);

  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Y));
  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  return output;
}

static int run_profile() {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const size_t rows = 2048;
  const size_t cols = 8192;
  const size_t total = rows * cols;
  const size_t bytes = total * sizeof(float);
  const auto input = make_rms_norm_input(rows, cols);
  float* d_X = nullptr;
  float* d_Y = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaMalloc(&d_X, bytes));
  CUDA_CHECK(cudaMalloc(&d_Y, bytes));
  CUDA_CHECK(cudaMemcpy(d_X, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < kProfileWarmupIterations; ++i) {
    solution(d_X, d_Y, rows, cols);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kProfileIterations; ++i) {
    solution(d_X, d_Y, rows, cols);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  const double device_mib =
      static_cast<double>(bytes * 2) / (1024.0 * 1024.0);
  std::cout << std::fixed << std::setprecision(3)
            << "profile scope=kernel-only verify=off"
            << " kernel=" << current_kernel_name()
            << " rows=" << rows << " cols=" << cols
            << " block_x=" << g_launch_config.block_x
            << " grid_x=" << g_launch_config.grid_x
            << " warmup=" << kProfileWarmupIterations
            << " repeats=" << kProfileIterations
            << " device_mib=" << device_mib
            << " avg_kernel_ms=" << elapsed_ms / kProfileIterations << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Y));
  return 0;
}

static int run_tests(bool skip_cpu_verify) {
  std::vector<TestResult> results;
  bool all_ok = true;
  const LaunchConfig default_launch = g_launch_config;

  const std::vector<TestCase> exact_tests = {
      {"sample_2x4",
       2,
       4,
       {1.0f, -1.0f, 2.0f, -2.0f,
        3.0f, 4.0f, 0.0f, -5.0f},
       {0.63245427f, -0.63245427f, 1.26490853f, -1.26490853f,
        0.84852780f, 1.13137040f, 0.0f, -1.41421300f}},
      {"tail_2x3",
       2,
       3,
       {0.0f, 0.0f, 0.0f,
        2.0f, -2.0f, 1.0f},
       {0.0f, 0.0f, 0.0f,
        1.15469861f, -1.15469861f, 0.57734931f}},
  };

  const std::vector<SizedCase> medium_tests = {
      {"medium_1", 8, 8},
      {"medium_rect", 17, 33},
      {"medium_tail", 31, 65},
  };

  const std::vector<SizedCase> large_tests = {
      {"large_1", 257, 513},
      {"large_2", 511, 1025},
  };

  const std::vector<SizedCase> tensara_tests = {
      {"tensara_1", 1024, 1024},
      {"tensara_2", 1024, 4096},
      {"tensara_3", 2048, 8192},
      {"tensara_4", 512, 16384},
  };

  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};
  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } scale_tests[] = {
      {"scale_tensara_1", 1024, 1024},
      {"scale_tensara_2", 1024, 4096},
      {"scale_tensara_3", 2048, 8192},
      {"scale_tensara_4", 512, 16384},
  };

  const KernelVariant kernel_variants[] = {
      KernelVariant::kBasic,
      KernelVariant::kFloat4,
      KernelVariant::kSharedMem,
      KernelVariant::kWarp,
  };

  if (kGpuKernelImplemented && !cuda_runtime_ready()) {
    return 1;
  }

  auto record_case = [&](const std::string& group, const std::string& name,
                         const std::vector<float>& input,
                         const std::vector<float>* expected, size_t rows,
                         size_t cols) {
    TestResult res;
    res.group = group;
    res.name = name;
    res.kernel = current_kernel_name();
    res.rows = rows;
    res.cols = cols;
    res.block_x = g_launch_config.block_x;
    res.grid_x = g_launch_config.grid_x;
    res.cpu = "SKIP";
    res.gpu = "SKIP";
    g_last_timing = {};

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      std::vector<float> ref(rows * cols, 0.0f);
      cpu_rms_norm_reference(input, ref, rows, cols);
      res.cpu = "REF";
      if (expected != nullptr) {
        const bool cpu_ok = verify_close(ref, *expected, 1e-4f, 2e-4f,
                                         name.c_str(), false);
        res.cpu = cpu_ok ? "PASS" : "FAIL";
        all_ok &= cpu_ok;
      }
    }

    if (kGpuKernelImplemented) {
      const auto gpu = run_gpu_case(input, rows, cols);
      if (expected != nullptr) {
        const bool gpu_ok = verify_close(gpu, *expected, 1e-4f, 2e-4f,
                                         name.c_str(), false);
        res.gpu = gpu_ok ? "PASS" : "FAIL";
        all_ok &= gpu_ok;
      } else if (!skip_cpu_verify && kCpuReferenceImplemented) {
        std::vector<float> ref(rows * cols, 0.0f);
        cpu_rms_norm_reference(input, ref, rows, cols);
        const bool gpu_ok = verify_close(gpu, ref, 1e-4f, 2e-4f,
                                         name.c_str(), false);
        res.gpu = gpu_ok ? "PASS" : "FAIL";
        all_ok &= gpu_ok;
      } else {
        res.gpu = "SKIP";
      }
    }

    res.total_ms = g_last_timing.total_ms;
    res.kernel_ms = g_last_timing.kernel_ms;
    results.push_back(res);
  };

  for (KernelVariant kernel_variant : kernel_variants) {
    if (!kernel_enabled(kernel_variant)) {
      continue;
    }
    g_kernel_variant = kernel_variant;
    g_launch_config = default_launch;

    for (const auto& tc : exact_tests) {
      record_case("small", tc.name, tc.input, &tc.expected, tc.rows,
                  tc.cols);
    }

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      for (const auto& tc : medium_tests) {
        const std::vector<float> input =
            make_rms_norm_input(tc.rows, tc.cols);
        record_case("medium", tc.name, input, nullptr, tc.rows, tc.cols);
      }
      for (const auto& tc : large_tests) {
        const std::vector<float> input =
            make_rms_norm_input(tc.rows, tc.cols);
        record_case("large", tc.name, input, nullptr, tc.rows, tc.cols);
      }
    }

    if (skip_cpu_verify) {
      for (const auto& tc : tensara_tests) {
        const std::vector<float> input =
            make_rms_norm_input(tc.rows, tc.cols);
        record_case("tensara", tc.name, input, nullptr, tc.rows, tc.cols);
      }
      for (const auto& tc : scale_tests) {
        const std::vector<float> input =
            make_rms_norm_input(tc.rows, tc.cols);
        for (int block_x : scale_block_sizes) {
          for (int grid_x : scale_grid_sizes) {
            g_launch_config = LaunchConfig{block_x, grid_x};
            record_case("scale", tc.name, input, nullptr, tc.rows,
                        tc.cols);
          }
        }
      }
    }
  }

  g_kernel_variant = KernelVariant::kBasic;
  g_launch_config = default_launch;
  std::cout << "Timing samples: mode=" << timing_mode_name()
            << " repeats=" << g_timing_repeats
            << " warmup=" << kTimingWarmupIterations
            << " metric=kernel_ms\n";
  std::cout << "CPU reference implemented: "
            << (kCpuReferenceImplemented ? "yes" : "no") << '\n';
  std::cout << "GPU kernel implemented: "
            << (kGpuKernelImplemented ? "yes" : "no") << "\n\n";

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
    const std::string arg = argv[i];
    if (arg == "--skip-cpu") {
      skip_cpu_verify = true;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [--skip-cpu] [--profile] [--kernel=NAME]"
                << " [--timing=median|best] [--timing-repeats=N]"
                << " [--list-kernels]\n";
      return 0;
    } else if (arg == "--list-kernels") {
      std::cout << "basic\nfloat4\nshared_mem\nwarp\n";
      return 0;
    } else if (arg == "--profile") {
      profile_mode = true;
    } else if (arg.rfind("--kernel=", 0) == 0) {
      if (!parse_kernel_arg(arg)) {
        return 1;
      }
    } else if (arg.rfind("--timing=", 0) == 0) {
      if (!parse_timing_arg(arg)) {
        return 1;
      }
    } else if (arg.rfind("--timing-repeats=", 0) == 0) {
      if (!parse_timing_repeats_arg(arg)) {
        return 1;
      }
    } else {
      std::cerr << "Unknown argument: " << arg
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
