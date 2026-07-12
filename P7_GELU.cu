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
 * Problem 7: GELU
 * Source: https://tensara.org/problems/gelu
 *
 * Given an input matrix A with shape M x N, compute matrix C with the same
 * shape by applying the Gaussian Error Linear Unit activation elementwise.
 *
 *   C[i][j] = GELU(A[i][j])
 *
 * Tensara asks for the tanh approximation:
 *
 *             x /      /  sqrt(2)                         \\
 *   GELU(x) = - | 1 + tanh| -------- * (x + 0.044715*x^3) | |
 *             2 \      \  sqrt(pi)                        / /
 *
 * Equivalent common form:
 *
 *   GELU(x) = 0.5*x *
 *             (1 + tanh(sqrt(2/pi) * (x + 0.044715*x*x*x)))
 *
 * Input/output shape rules:
 * - input is a row-major float32 matrix with shape M x N.
 * - output is a row-major float32 matrix with shape M x N.
 * - The published problem definition passes extra params as [M, N].
 * - The local solution signature below therefore treats n as rows M and
 *   m as columns N.
 *
 * Important notes:
 * - Implement the approximation formula above, not the exact normal CDF.
 * - The problem definition verifies with rtol=1e-4 and atol=2e-5.
 * - The static page did not display starter code, but embedded problem data
 *   confirmed the parameter order: input, output, n, m.
 *
 * Published Tensara sizes:
 * - 4096x4096
 * - 6144x4096
 * - 4096x7168
 * - 4096x8192
 * - 8192x8192
 */

// Tensara-style signature:
// - input and output are device pointers
// - n is rows M from the problem definition
// - m is columns N from the problem definition
// - input/output are row-major matrices with shape (n, m)
extern "C" void solution(const float* input, float* output, size_t n,
                         size_t m);

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

enum class TimingMode {
  kMedian,
  kBest,
};

enum class KernelVariant {
  kBasic,
  kFloat4,
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

  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic or --kernel=float4)\n";
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

// CPU reference implementation.
//
// input: row-major matrix flattened with shape (rows * cols)
// output: row-major matrix flattened with shape (rows * cols)
// rows: matrix row count M
// cols: matrix column count N
static void cpu_gelu_reference(const std::vector<float>& input,
                               std::vector<float>& output, size_t rows,
                               size_t cols) {
  size_t total = rows * cols;

  for (size_t i = 0; i < total; ++i) {
    float x = input[i];
    float internal = (0.7978845608f * (x + (0.044715f * x * x * x) ) );
    output[i] = 0.5f * x * (1.0f + std::tanh(internal));
  }
}


//shared device helpers
__device__ float gelu_single(const float x)
{
  float x_int = (0.7978845608f * (x + (0.044715f * x * x * x) ) );
  float y     = 0.5f * x * (1.0f + std::tanh(x_int));
  return y;
}

// Basic GPU kernel implementation.
//
// input: device pointer to row-major matrix with shape (total)
// output: device pointer to row-major matrix with shape (total)
// total: number of matrix elements, normally rows * cols from solution(...)
__global__ void gelu_basic_kernel(const float* input, float* output,
                                  size_t total) {

  size_t gix = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gix; gx < total; gx += grid_stride)
    output[gx] = gelu_single(input[gx]);
}

// Float4 GPU kernel implementation.
//
// input: device pointer to row-major matrix with shape (total)
// output: device pointer to row-major matrix with shape (total)
// total: number of matrix elements, normally rows * cols from solution(...)
__global__ void gelu_float4_kernel(const float* input, float* output,
                                   size_t total) {

  size_t total_f4 = total / 4;
  const float4 *input_f4 = reinterpret_cast<const float4 *>(input);
  float4 *output_f4 = reinterpret_cast<float4 *>(output);
  size_t gix = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  //vector part
  for (size_t gx = gix; gx < total_f4; gx += grid_stride)
  {
    float4 x_f4 = input_f4[gx];
    float4 y_f4;
    y_f4.w = gelu_single(x_f4.w);
    y_f4.x = gelu_single(x_f4.x);
    y_f4.y = gelu_single(x_f4.y);
    y_f4.z = gelu_single(x_f4.z);
    output_f4[gx] = y_f4;
  }

  //scalar/tail part
  size_t tail_start = (total_f4 * 4) + gix;
  for (size_t gx = tail_start; gx < total; gx += grid_stride)
    output[gx] = gelu_single(input[gx]);
}

extern "C" void solution(const float* input, float* output, size_t n,
                         size_t m) {
  const size_t total = n * m;
  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      gelu_basic_kernel<<<grid_shape, block_shape>>>(input, output, total);
      break;
    case KernelVariant::kFloat4:
      gelu_float4_kernel<<<grid_shape, block_shape>>>(input, output, total);
      break;
  }

  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_gelu_input(size_t rows, size_t cols) {
  const size_t total = rows * cols;
  std::vector<float> input(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 41 + rows * 13 + cols * 5 + 23) % 257) - 128;
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
    std::cerr << "verify(" << label << "): PASS max_abs=" << max_abs
              << " max_i=" << max_i << '\n';
  }
  return true;
}

struct TestResult {
  const char* group = "";
  const char* name = "";
  const char* kernel = "";
  size_t rows = 0;
  size_t cols = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group" << std::setw(18) << "name"
            << std::setw(12) << "kernel" << std::setw(10) << "rows"
            << std::setw(10) << "cols" << std::setw(8) << "block_x"
            << std::setw(8) << "grid_x" << std::setw(6) << "cpu"
            << std::setw(6) << "gpu" << std::setw(12) << "total_ms"
            << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(110, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& r : results) {
    std::cout << std::left << std::setw(8) << r.group << std::setw(18)
              << r.name << std::setw(12) << r.kernel << std::setw(10)
              << r.rows << std::setw(10) << r.cols << std::setw(8)
              << r.block_x << std::setw(8) << r.grid_x << std::setw(6)
              << r.cpu << std::setw(6) << r.gpu << std::setw(12)
              << r.total_ms << std::setw(12) << r.kernel_ms << '\n';
  }
}

static void print_scale_heatmaps(const std::vector<TestResult>& results) {
  if (!kGpuKernelImplemented) {
    return;
  }

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

      std::cout << '\n' << name << " / " << kernel << " best=(" << best_block
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

static void collect_kernel_timing_samples(const float* d_input,
                                          float* d_output, size_t bytes,
                                          size_t rows, size_t cols,
                                          cudaEvent_t kernel_start,
                                          cudaEvent_t kernel_stop,
                                          std::vector<float>& samples) {
  for (int i = 0; i < kTimingWarmupIterations; ++i) {
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));
    solution(d_input, d_output, rows, cols);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  samples.clear();
  samples.reserve(g_timing_repeats);
  for (int i = 0; i < g_timing_repeats; ++i) {
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    solution(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    samples.push_back(kernel_ms);
  }
}

static void run_solution_host(const std::vector<float>& input,
                              std::vector<float>& output, size_t rows,
                              size_t cols) {
  cudaEvent_t total_start = nullptr;
  cudaEvent_t total_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  const size_t total = rows * cols;
  const size_t bytes = total * sizeof(float);

  float* d_input = nullptr;
  float* d_output = nullptr;

  CUDA_CHECK(cudaEventRecord(total_start));
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(
      cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, bytes));

  std::vector<float> kernel_samples;
  collect_kernel_timing_samples(d_input, d_output, bytes, rows, cols,
                                kernel_start, kernel_stop, kernel_samples);

  CUDA_CHECK(
      cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
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
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

static int run_profile() {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const size_t rows = 6144;
  const size_t cols = 4096;
  const size_t total = rows * cols;
  const size_t bytes = total * sizeof(float);
  const auto input = make_gelu_input(rows, cols);
  float* d_input = nullptr;
  float* d_output = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < kProfileWarmupIterations; ++i) {
    solution(d_input, d_output, rows, cols);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kProfileIterations; ++i) {
    solution(d_input, d_output, rows, cols);
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
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return 0;
}

static int run_tests(bool skip_cpu_verify) {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const LaunchConfig default_launch = g_launch_config;
  const std::vector<TestCase> small_tests = {
      {"small_1",
       3,
       3,
       {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f},
       {-0.00363739f, -0.04540231f, -0.15880801f, -0.15428599f,
        0.0f, 0.34571401f, 0.84119199f, 1.95459769f, 2.99636261f}},
      {"small_2",
       2,
       3,
       {-4.0f, 4.0f, 0.25f, -0.25f, 1.0f, -1.0f},
       {-0.00007025f, 3.99992975f, 0.14967535f, -0.10032465f,
        0.84119199f, -0.15880801f}},
      {"small_tail",
       1,
       5,
       {-1.25f, -0.75f, 0.125f, 2.5f, -2.5f},
       {-0.13228580f, -0.17003944f, 0.06871720f, 2.48491573f,
        -0.01508427f}},
  };

  const struct {
    const char* name;
    const char* scale_name;
    size_t rows;
    size_t cols;
  } tensara_tests[] = {
      {"tensara_1", "scale_tensara_1", 4096, 4096},
      {"tensara_2", "scale_tensara_2", 6144, 4096},
      {"tensara_3", "scale_tensara_3", 4096, 7168},
      {"tensara_4", "scale_tensara_4", 4096, 8192},
      {"tensara_5", "scale_tensara_5", 8192, 8192},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } medium_tests[] = {
      {"medium_1", 64, 64},
      {"medium_2", 255, 257},
      {"medium_rect", 512, 1025},
      {"medium_tail", 257, 258},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } large_verify_tests[] = {
      {"large_1", 1023, 2049},
      {"large_2", 1537, 2049},
  };

  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};
  const KernelVariant kernel_variants[] = {KernelVariant::kBasic,
                                           KernelVariant::kFloat4};

  bool all_ok = true;
  std::vector<TestResult> results;

  auto run_case = [&](const char* group, const char* name,
                      const std::vector<float>& input,
                      const std::vector<float>* expected, size_t rows,
                      size_t cols) {
    std::vector<float> ref(input.size(), 0.0f);
    std::vector<float> gpu_out(input.size(), 0.0f);
    run_solution_host(input, gpu_out, rows, cols);

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

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      cpu_gelu_reference(input, ref, rows, cols);
      res.cpu = "REF";

      if (expected != nullptr) {
        const bool cpu_ok =
            verify_close(ref, *expected, 2e-5f, 1e-4f, name, false);
        all_ok &= cpu_ok;
        res.cpu = cpu_ok ? "PASS" : "FAIL";
      }

      if (kGpuKernelImplemented) {
        const bool gpu_ok =
            verify_close(gpu_out, ref, 2e-5f, 1e-4f, name, false);
        all_ok &= gpu_ok;
        res.gpu = gpu_ok ? "PASS" : "FAIL";
      }
    } else if (expected != nullptr && kGpuKernelImplemented) {
      const bool gpu_ok =
          verify_close(gpu_out, *expected, 2e-5f, 1e-4f, name, false);
      all_ok &= gpu_ok;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
    }

    res.total_ms = g_last_timing.total_ms;
    res.kernel_ms = g_last_timing.kernel_ms;
    results.push_back(res);
  };

  auto run_sized = [&](const char* group, const char* name, size_t rows,
                       size_t cols) {
    g_launch_config = default_launch;
    const auto input = make_gelu_input(rows, cols);
    run_case(group, name, input, nullptr, rows, cols);
  };

  auto run_scaling = [&](const char* name, size_t rows, size_t cols) {
    const auto input = make_gelu_input(rows, cols);
    for (int block_x : scale_block_sizes) {
      for (int grid_x : scale_grid_sizes) {
        g_launch_config = {block_x, grid_x};
        run_case("scale", name, input, nullptr, rows, cols);
      }
    }
  };

  for (KernelVariant kernel_variant : kernel_variants) {
    if (!kernel_enabled(kernel_variant)) {
      continue;
    }
    g_kernel_variant = kernel_variant;
    g_launch_config = default_launch;

    for (const auto& tc : small_tests) {
      run_case("small", tc.name, tc.input, &tc.expected, tc.rows, tc.cols);
    }

    if (!skip_cpu_verify) {
      for (const auto& mt : medium_tests) {
        run_sized("medium", mt.name, mt.rows, mt.cols);
      }
      for (const auto& lt : large_verify_tests) {
        run_sized("large", lt.name, lt.rows, lt.cols);
      }
    }

    if (skip_cpu_verify) {
      for (const auto& tt : tensara_tests) {
        run_sized("tensara", tt.name, tt.rows, tt.cols);
      }
      for (const auto& tt : tensara_tests) {
        run_scaling(tt.scale_name, tt.rows, tt.cols);
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
      std::cout << "basic\nfloat4\n";
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
                << "--timing=..., "
                << "--timing-repeats=...)\n";
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
