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
 * Problem 6: 1D Average Pooling
 * Source: https://tensara.org/problems/avg-pool-1d
 *
 * Given an input vector of length H, compute average pooling with window
 * length k, stride S, and symmetric zero padding P.
 *
 *              1   k - 1
 * output[i] = ---  sum   input[S * i + m - P]
 *              k   m = 0
 *
 * Out-of-range input positions contribute zero, but the divisor is always the
 * full kernel size k. The output length is:
 *
 *         / H + 2 * P - k     \
 * H_out =| ------------- + 1  |
 *         \       S           /
 *
 * where the division is integer floor division.
 *
 * Published Tensara sizes:
 * - H=2097152, K=7, S=4, P=3
 * - H=4194304, K=2, S=1, P=0
 * - H=8388608, K=3, S=2, P=1
 * - H=16777216, K=4, S=2, P=1
 * - H=33554432, K=3, S=1, P=1
 * - H=67108864, K=5, S=3, P=2
 */

// Tensara-style signature:
// - input and output are device pointers to float32 arrays
// - input has length H
// - output has length H_out from the formula above
// - kernel_size, stride, and padding are scalar pooling parameters
extern "C" void solution(const float* input, int kernel_size, int stride,
                         int padding, float* output, size_t H);

static constexpr bool kCpuReferenceImplemented = true;
static constexpr bool kGpuKernelImplemented = true;
static constexpr int kDefaultTimingRepeats = 5;
static constexpr int kTimingWarmupIterations = 1;

struct Timing {
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

struct LaunchConfig {
  int block_x = 256;
  int grid_x = 64;
};

enum class TimingMode {
  kMedian,
  kBest,
};

enum class KernelVariant {
  kBasic,
  kBasicLdg,
  kCoopShared,
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
    case KernelVariant::kBasicLdg:
      return "basic_ldg";
    case KernelVariant::kCoopShared:
      return "coop_shared";
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
  if (value == "basic_ldg") {
    g_kernel_variant = KernelVariant::kBasicLdg;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "coop_shared" || value == "shared") {
    g_kernel_variant = KernelVariant::kCoopShared;
    g_kernel_arg_set = true;
    return true;
  }

  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic, --kernel=basic_ldg, "
            << "or --kernel=coop_shared)\n";
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

static size_t avg_pool_output_size(size_t H, int kernel_size, int stride,
                                   int padding) {
  if (kernel_size <= 0 || stride <= 0 || padding < 0) {
    return 0;
  }

  const long long padded =
      static_cast<long long>(H) + (2LL * static_cast<long long>(padding));
  if (padded < static_cast<long long>(kernel_size)) {
    return 0;
  }

  return static_cast<size_t>(
      ((padded - static_cast<long long>(kernel_size)) / stride) + 1);
}

struct TestCase {
  const char* group = "";
  const char* name = "";
  size_t H = 0;
  int kernel_size = 0;
  int stride = 0;
  int padding = 0;
  std::vector<float> input;
  std::vector<float> expected;
};

struct TestResult {
  std::string group;
  std::string name;
  std::string kernel;
  size_t H = 0;
  size_t H_out = 0;
  int kernel_size = 0;
  int stride = 0;
  int padding = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

// Canonical CPU reference stub.
//
// input: host vector with length H
// kernel_size: pooling window length
// stride: distance between consecutive window starts
// padding: implicit zero-padding count on each side of input
// output: host vector with length H_out
// H: input vector length
static void cpu_avg_pool_1d_cannonical(const std::vector<float>& input,
                                       int kernel_size, int stride,
                                       int padding,
                                       std::vector<float>& output, size_t H) {
  //output centric
  size_t gx = 0;
  const size_t N = output.size();

  //for all output elements
  for (gx = 0; gx < N; ++gx)
  {
    //compute window start and end (including padding)
    size_t ws = gx * stride;
    size_t we = ws + kernel_size;

    //map window start and end to actual indexes
    ws = (ws >= padding) ? (ws - padding) : 0; //left boundary, ignore right
    we = (we >= padding) ? (we - padding) : 0; //left boundary
    we = (we > H) ? H : we; //right boundary

    //accumulate
    float wsum = 0.0f;
    for (size_t i = ws; i < we; ++i)
      wsum += input[i];

    output[gx] = wsum / kernel_size;
  }
}

// Sliding-window CPU variant.
//
// input: host vector with length H
// kernel_size: pooling window length
// stride: distance between consecutive window starts
// padding: implicit zero-padding count on each side of input
// output: host vector with length H_out
// H: input vector length
[[maybe_unused]] static void cpu_avg_pool_1d_sliding_window(
    const std::vector<float>& input, int kernel_size, int stride, int padding,
    std::vector<float>& output, size_t H)
{
  size_t ws = 0;
  size_t we = 0;
  size_t total_size = H + (2 * padding);
  float wsum = 0.0f;
  size_t out_idx = 0;

  while (ws + kernel_size <= total_size)
  {
    //setup window limits
    size_t wl = ws + kernel_size;

    //sum up current window
    while (we < wl)
    {
      if (padding <= we && we < H + padding)
        wsum += input[we - padding];
      we++;
    }

    //store output
    output[out_idx++] = wsum / kernel_size;

    //setup next window start and end
    //if stride >= window size, discard window sum completely
    if (stride >= kernel_size)
    {
      wsum = 0.0f;
      ws += stride;
      we = ws;
    }
    //stride is less than window, can re-use parts of window sum
    else
    {
      //shrink to remove out of window sum
      size_t wshr = ws + stride;
      while (ws < wshr)
      {
        if (padding <= ws && ws < H + padding)
          wsum -= input[ws - padding];
        ws++;
      }
    }
  }
}

// Basic GPU kernel stub.
//
// input: device vector with length H
// kernel_size: pooling window length
// stride: distance between consecutive window starts
// padding: implicit zero-padding count on each side of input
// output: device vector with length H_out
// H: input vector length
// H_out: output vector length
__global__ void avg_pool_1d_basic_kernel(const float* input, int kernel_size,
                                         int stride, int padding,
                                         float* output, size_t H,
                                         size_t H_out) {
  size_t gix = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gix; gx < H_out; gx += grid_stride)
  {
    //input window start and end
    size_t ws = gx * stride;
    size_t we = ws + kernel_size;

    //valid input windows only
    ws = (ws >= padding) ? (ws - padding) : 0; //left guard
    we = (we >= padding) ? (we - padding) : 0; //left guard
    we = (we > H) ? H : we; //right guard

    //compute window sum
    float wsum = 0.0f;
    for (size_t i = ws; i < we; ++i)
      wsum += input[i];

    output[gx] = wsum / kernel_size;
  }
}

// Basic GPU kernel with restricted pointers and read-only input loads.
//
// input: device vector with length H
// kernel_size: pooling window length
// stride: distance between consecutive window starts
// padding: implicit zero-padding count on each side of input
// output: device vector with length H_out
// H: input vector length
// H_out: output vector length
__global__ void avg_pool_1d_basic_ldg_kernel(
    const float* __restrict__ input, int kernel_size, int stride, int padding,
    float* __restrict__ output, size_t H, size_t H_out) {
  size_t gix = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gix; gx < H_out; gx += grid_stride)
  {
    //input window start and end
    size_t ws = gx * stride;
    size_t we = ws + kernel_size;

    //valid input windows only
    ws = (ws >= padding) ? (ws - padding) : 0; //left guard
    we = (we >= padding) ? (we - padding) : 0; //left guard
    we = (we > H) ? H : we; //right guard

    //compute window sum
    float wsum = 0.0f;
    for (size_t i = ws; i < we; ++i)
      wsum += __ldg(&input[i]);

    output[gx] = wsum / kernel_size;
  }
}

// Cooperative shared-memory GPU kernel stub.
//
// input: device vector with length H
// kernel_size: pooling window length
// stride: distance between consecutive window starts
// padding: implicit zero-padding count on each side of input
// output: device vector with length H_out
// H: input vector length
// H_out: output vector length
__global__ void avg_pool_1d_coop_shared_kernel(
    const float* __restrict__ input, int kernel_size, int stride, int padding,
    float* __restrict__ output, size_t H, size_t H_out) {
  //shared memory to hold up that many elements, size passed in dynamically
  extern __shared__ float smem_input[];

  size_t total_blocks = (H_out + blockDim.x - 1) / blockDim.x;

  //grid stride @ block level
  for (size_t bx = blockIdx.x; bx < total_blocks; bx += gridDim.x)
  {
    //output cells range owned by a block = blockDim.x
    //compute the output cell, start end end range for this block
    size_t op_start = bx * blockDim.x;
    size_t op_end = op_start + blockDim.x - 1;

    //need to compute which portion of input cells are required for a block to process
    size_t input_start = op_start * stride; //ech output elements starts/skip over stride num of elemnts
    //the last output cell start with a certain stride and spans over kenrl_Size worh of elements
    size_t input_end = (op_end * stride) + kernel_size;
    size_t total_input_elements = input_end - input_start;

    //populate shared memory cooperativetly
    for (size_t lx = threadIdx.x; lx < total_input_elements; lx += blockDim.x)
    {
      long long load_idx =
          static_cast<long long>(input_start + lx) - padding;
      smem_input[lx] =
          (0 <= load_idx && load_idx < static_cast<long long>(H))
              ? input[load_idx]
              : 0.0f;
    }

    //sync
    __syncthreads();

    //perform average pooling on shared memory
    float sum = 0.0f;
    size_t local_ws = threadIdx.x * stride;
    size_t local_we = local_ws + kernel_size;
    for (size_t i = local_ws; i < local_we; ++i)
      sum += smem_input[i];

    //update output
    size_t out_idx = (bx * blockDim.x) + threadIdx.x;

    if (out_idx < H_out)
      output[out_idx] = sum / kernel_size;

    //make sure current grid is entirely processed, before
    //moving to next grid
    __syncthreads();
  }
}

extern "C" void solution(const float* input, int kernel_size, int stride,
                         int padding, float* output, size_t H) {
  const size_t H_out = avg_pool_output_size(H, kernel_size, stride, padding);

  if (H_out == 0) {
    return;
  }

  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic: {
      avg_pool_1d_basic_kernel<<<grid_shape, block_shape>>>(
          input, kernel_size, stride, padding, output, H, H_out);
      break;
    }
    case KernelVariant::kBasicLdg: {
      avg_pool_1d_basic_ldg_kernel<<<grid_shape, block_shape>>>(
          input, kernel_size, stride, padding, output, H, H_out);
      break;
    }
    case KernelVariant::kCoopShared: {
      size_t output_cells_per_block = block_shape.x - 1;
      size_t input_cells_per_block =
          output_cells_per_block * stride + kernel_size;
      size_t smem_bytes = input_cells_per_block * sizeof(float);
      avg_pool_1d_coop_shared_kernel<<<grid_shape, block_shape, smem_bytes>>>(
          input, kernel_size, stride, padding, output, H, H_out);
      break;
    }
  }

  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_avg_pool_input(size_t H) {
  std::vector<float> input(H, 0.0f);
  for (size_t i = 0; i < H; ++i) {
    const int raw = static_cast<int>((i * 31 + H * 17 + 7) % 251) - 125;
    input[i] = static_cast<float>(raw) / 37.0f;
  }
  return input;
}

static bool verify_close(const std::vector<float>& got,
                         const std::vector<float>& expected, float atol,
                         float rtol, const char* label) {
  if (got.size() != expected.size()) {
    std::cerr << "verify(" << label << "): size mismatch got="
              << got.size() << " expected=" << expected.size() << '\n';
    return false;
  }

  float max_abs = 0.0f;
  size_t max_i = 0;
  bool ok = true;
  size_t first_bad = 0;

  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) {
      if (ok) {
        first_bad = i;
      }
      ok = false;
      continue;
    }

    const float diff = std::fabs(got[i] - expected[i]);
    const float tol = atol + rtol * std::fabs(expected[i]);
    if (diff > max_abs) {
      max_abs = diff;
      max_i = i;
    }
    if (diff > tol) {
      if (ok) {
        first_bad = i;
      }
      ok = false;
    }
  }

  if (!ok) {
    std::cerr << std::fixed << std::setprecision(6)
              << "verify(" << label << "): first_bad=" << first_bad
              << " got=" << got[first_bad]
              << " expected=" << expected[first_bad]
              << " max_abs=" << max_abs
              << " max_i=" << max_i << '\n';
  }

  return ok;
}

static Timing run_kernel_timed(float* d_input, int kernel_size, int stride,
                               int padding, float* d_output, size_t H,
                               cudaEvent_t kernel_start,
                               cudaEvent_t kernel_stop) {
  std::vector<float> kernel_samples;
  kernel_samples.reserve(g_timing_repeats);

  for (int i = 0; i < kTimingWarmupIterations; ++i) {
    solution(d_input, kernel_size, stride, padding, d_output, H);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < g_timing_repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(kernel_start));
    solution(d_input, kernel_size, stride, padding, d_output, H);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    kernel_samples.push_back(kernel_ms);
  }

  Timing timing;
  timing.kernel_ms = select_timing_sample(kernel_samples);
  return timing;
}

static void run_solution_host(const std::vector<float>& input,
                              int kernel_size, int stride, int padding,
                              std::vector<float>& output, size_t H) {
  const size_t H_out = output.size();
  if (H_out == 0) {
    g_last_timing = Timing{};
    return;
  }

  const size_t input_bytes = input.size() * sizeof(float);
  const size_t output_bytes = output.size() * sizeof(float);

  float* d_input = nullptr;
  float* d_output = nullptr;
  cudaEvent_t total_start = nullptr;
  cudaEvent_t total_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;

  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  CUDA_CHECK(cudaEventRecord(total_start));
  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

  Timing timing = run_kernel_timed(d_input, kernel_size, stride, padding,
                                   d_output, H, kernel_start, kernel_stop);

  if (H_out > 0) {
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes,
                          cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));
  timing.total_ms = total_ms;
  g_last_timing = timing;

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group"
            << std::setw(18) << "name"
            << std::setw(12) << "kernel"
            << std::right << std::setw(12) << "H"
            << std::setw(12) << "H_out"
            << std::setw(8) << "K"
            << std::setw(8) << "S"
            << std::setw(8) << "P"
            << std::setw(8) << "block"
            << std::setw(8) << "grid"
            << std::setw(7) << "cpu"
            << std::setw(7) << "gpu"
            << std::setw(12) << "total_ms"
            << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(140, '-') << '\n';

  for (const auto& res : results) {
    std::cout << std::left << std::setw(8) << res.group
              << std::setw(18) << res.name
              << std::setw(12) << res.kernel
              << std::right << std::setw(12) << res.H
              << std::setw(12) << res.H_out
              << std::setw(8) << res.kernel_size
              << std::setw(8) << res.stride
              << std::setw(8) << res.padding
              << std::setw(8) << res.block_x
              << std::setw(8) << res.grid_x
              << std::setw(7) << res.cpu
              << std::setw(7) << res.gpu
              << std::setw(12) << std::fixed << std::setprecision(3)
              << res.total_ms
              << std::setw(12) << std::fixed << std::setprecision(3)
              << res.kernel_ms << '\n';
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
  std::cout << "\nScale heatmaps: kernel_ms, '-' means no row\n";

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

static int run_tests(bool skip_cpu_verify) {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const TestCase small_tests[] = {
      {"small", "pad_stride_1", 5, 3, 1, 1,
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
       {1.0f, 2.0f, 3.0f, 4.0f, 3.0f}},
      {"small", "stride_2", 6, 2, 2, 0,
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
       {1.5f, 3.5f, 5.5f}},
      {"small", "wide_pad", 4, 4, 1, 2,
       {2.0f, 4.0f, 6.0f, 8.0f},
       {1.5f, 3.0f, 5.0f, 4.5f, 3.5f}},
      {"small", "kernel_1", 4, 1, 1, 0,
       {-1.0f, 0.0f, 2.5f, -3.5f},
       {-1.0f, 0.0f, 2.5f, -3.5f}},
      {"small", "pad_stride_2", 3, 3, 2, 2,
       {6.0f, 9.0f, 12.0f},
       {2.0f, 9.0f, 4.0f}},
      {"small", "stride_gap", 7, 2, 3, 0,
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
       {1.5f, 4.5f}},
      {"small", "pad_zero_windows", 1, 2, 1, 3,
       {4.0f},
       {0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f}},
      {"small", "zero_output", 2, 5, 1, 0,
       {1.0f, 2.0f},
       {}},
  };

  const struct {
    const char* name;
    size_t H;
    int kernel_size;
    int stride;
    int padding;
  } medium_tests[] = {
      {"sample", 16, 3, 2, 1},
      {"medium_1", 4096, 7, 4, 3},
      {"medium_2", 65536, 3, 2, 1},
      {"medium_3", 262144, 5, 3, 2},
  };

  const struct {
    const char* name;
    size_t H;
    int kernel_size;
    int stride;
    int padding;
  } large_verify_tests[] = {
      {"large_1", 1048576, 7, 4, 3},
      {"large_2", 1048583, 5, 3, 2},
  };

  const struct {
    const char* name;
    size_t H;
    int kernel_size;
    int stride;
    int padding;
  } scale_tests[] = {
      {"scale_tiny", 33, 3, 1, 1},
      {"scale_stride_gap", 513, 4, 5, 2},
      {"scale_tail", 4097, 7, 4, 3},
      {"scale_kernel_1", 1024, 1, 1, 0},
  };

  const int block_x_values[] = {64, 128, 256, 512};
  const int grid_x_values[] = {8, 16, 32, 64, 128};
  const KernelVariant kernel_variants[] = {
      KernelVariant::kBasic,
      KernelVariant::kBasicLdg,
      KernelVariant::kCoopShared,
  };

  const struct {
    const char* name;
    size_t H;
    int kernel_size;
    int stride;
    int padding;
  } tensara_tests[] = {
      {"tensara_1", 2097152, 7, 4, 3},
      {"tensara_2", 4194304, 2, 1, 0},
      {"tensara_3", 8388608, 3, 2, 1},
      {"tensara_4", 16777216, 4, 2, 1},
      {"tensara_5", 33554432, 3, 1, 1},
      {"tensara_6", 67108864, 5, 3, 2},
  };

  bool all_ok = true;
  std::vector<TestResult> results;
  const LaunchConfig default_launch_config = g_launch_config;

  auto run_case = [&](const char* group, const char* name,
                      const std::vector<float>& input, int kernel_size,
                      int stride, int padding,
                      const std::vector<float>* expected) {
    const size_t H = input.size();
    const size_t H_out = avg_pool_output_size(H, kernel_size, stride,
                                              padding);
    std::vector<float> gpu_out(H_out, 0.0f);
    run_solution_host(input, kernel_size, stride, padding, gpu_out, H);

    TestResult res;
    res.group = group;
    res.name = name;
    res.kernel = current_kernel_name();
    res.H = H;
    res.H_out = H_out;
    res.kernel_size = kernel_size;
    res.stride = stride;
    res.padding = padding;
    res.block_x = g_launch_config.block_x;
    res.grid_x = g_launch_config.grid_x;
    res.cpu = "SKIP";
    res.gpu = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      std::vector<float> ref(H_out, 0.0f);
      cpu_avg_pool_1d_cannonical(input, kernel_size, stride, padding, ref, H);
      res.cpu = "REF";

      if (expected != nullptr) {
        const bool cpu_ok =
            verify_close(ref, *expected, 1e-5f, 1e-5f, name);
        all_ok &= cpu_ok;
        res.cpu = cpu_ok ? "PASS" : "FAIL";
      }

      if (kGpuKernelImplemented) {
        const bool gpu_ok =
            verify_close(gpu_out, ref, 1e-4f, 1e-4f, name);
        all_ok &= gpu_ok;
        res.gpu = gpu_ok ? "PASS" : "FAIL";
      }
    } else if (expected != nullptr && kGpuKernelImplemented) {
      const bool gpu_ok =
          verify_close(gpu_out, *expected, 1e-5f, 1e-5f, name);
      all_ok &= gpu_ok;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
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
    g_launch_config = default_launch_config;

    for (const auto& tc : small_tests) {
      run_case(tc.group, tc.name, tc.input, tc.kernel_size, tc.stride,
               tc.padding, &tc.expected);
    }

    for (const auto& mt : medium_tests) {
      const auto input = make_avg_pool_input(mt.H);
      run_case("medium", mt.name, input, mt.kernel_size, mt.stride,
               mt.padding, nullptr);
    }

    for (const auto& lt : large_verify_tests) {
      const auto input = make_avg_pool_input(lt.H);
      run_case("large", lt.name, input, lt.kernel_size, lt.stride,
               lt.padding, nullptr);
    }

    if (!skip_cpu_verify) {
      for (const auto& st : scale_tests) {
        const auto input = make_avg_pool_input(st.H);
        for (const int block_x : block_x_values) {
          for (const int grid_x : grid_x_values) {
            g_launch_config = LaunchConfig{block_x, grid_x};
            run_case("scale", st.name, input, st.kernel_size, st.stride,
                     st.padding, nullptr);
          }
        }
      }
    }

    g_launch_config = default_launch_config;

    if (skip_cpu_verify) {
      for (const auto& tt : tensara_tests) {
        const auto input = make_avg_pool_input(tt.H);
        run_case("tensara", tt.name, input, tt.kernel_size, tt.stride,
                 tt.padding, nullptr);
      }
      for (const auto& tt : tensara_tests) {
        const auto input = make_avg_pool_input(tt.H);
        const std::string scale_name = std::string("scale_") + tt.name;
        for (const int block_x : block_x_values) {
          for (const int grid_x : grid_x_values) {
            g_launch_config = LaunchConfig{block_x, grid_x};
            run_case("scale", scale_name.c_str(), input, tt.kernel_size,
                     tt.stride, tt.padding, nullptr);
          }
        }
      }
    }
  }

  g_kernel_variant = KernelVariant::kBasic;
  g_launch_config = default_launch_config;

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
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--skip-cpu") {
      skip_cpu_verify = true;
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
                << " (supported: --skip-cpu, --kernel=..., --timing=..., "
                << "--timing-repeats=...)\n";
      return 1;
    }
  }

  return run_tests(skip_cpu_verify);
}
