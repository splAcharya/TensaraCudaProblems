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
 * Problem 8: Sigmoid
 * Source: https://tensara.org/problems/sigmoid
 *
 * Given an input matrix A with shape M x N, compute matrix C with the same
 * shape by applying the Sigmoid activation elementwise.
 *
 *   C[i][j] = sigmoid(A[i][j])
 *
 * Sigmoid is defined as:
 *
 *                1
 *   sigmoid(x) = ------------
 *                1 + exp(-x)
 *
 * Input/output shape rules:
 * - input is a row-major float32 matrix with shape M x N.
 * - output is a row-major float32 matrix with shape M x N.
 * - The published problem definition passes extra params as [M, N].
 * - The local solution signature below therefore treats n as rows M and
 *   m as columns N.
 *
 * Important notes:
 * - The problem definition verifies with rtol=1e-4 and atol=6e-5.
 * - The page and embedded problem data confirmed the parameter order:
 *   input, output, n, m.
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

// Shared host/device helper.
__host__ __device__ __forceinline__ float sigmoid_int(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

// CPU reference implementation.
//
// input: row-major matrix flattened with shape (rows * cols)
// output: row-major matrix flattened with shape (rows * cols)
// rows: matrix row count M
// cols: matrix column count N
static void cpu_sigmoid_reference(const std::vector<float>& input,
                                  std::vector<float>& output, size_t rows,
                                  size_t cols) {
  size_t total = rows * cols;

  for (size_t i = 0; i < total; ++i)
    output[i] = sigmoid_int(input[i]);
}

// Basic GPU kernel implementation.
//
// input: device pointer to row-major matrix with shape (total)
// output: device pointer to row-major matrix with shape (total)
// total: number of matrix elements, normally rows * cols from solution(...)
__global__ void sigmoid_basic_kernel(const float* input, float* output,
                                     size_t total) {
  size_t gix = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gix; gx < total; gx += grid_stride)
    output[gx] = sigmoid_int(input[gx]);
}

// Float4 GPU kernel implementation.
//
// input: device pointer to row-major matrix with shape (total)
// output: device pointer to row-major matrix with shape (total)
// total: number of matrix elements, normally rows * cols from solution(...)
// note: implementation should handle any scalar tail after float4 work
__global__ void sigmoid_float4_kernel(const float* input, float* output,
                                      size_t total) {

  size_t gix = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t total_f4 = total / 4;
  size_t tail_start = gix + (total_f4 * 4); //start from this many elemt apart
  size_t grid_stride = (blockDim.x * gridDim.x);

  const float4 *input_f4 = reinterpret_cast<const float4 *>(input);
  float4 *output_f4 = reinterpret_cast<float4 *>(output);

  for (size_t gx = gix; gx < total_f4; gx += grid_stride)
  {
    output_f4[gx] = make_float4(sigmoid_int(input_f4[gx].x),
                                sigmoid_int(input_f4[gx].y),
                                sigmoid_int(input_f4[gx].z),
                                sigmoid_int(input_f4[gx].w));
  }

  for (size_t gx = tail_start; gx < total; gx += grid_stride)
    output[gx] = sigmoid_int(input[gx]);
}

extern "C" void solution(const float* input, float* output, size_t n,
                         size_t m) {
  const size_t total = n * m;
  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      sigmoid_basic_kernel<<<grid_shape, block_shape>>>(input, output, total);
      break;
    case KernelVariant::kFloat4:
      sigmoid_float4_kernel<<<grid_shape, block_shape>>>(input, output,
                                                         total);
      break;
  }

  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_sigmoid_input(size_t rows, size_t cols) {
  const size_t total = rows * cols;
  std::vector<float> input(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 41 + rows * 13 + cols * 5 + 23) % 257) - 128;
    input[i] = static_cast<float>(raw) / 128.0f;
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
  samples.reserve(static_cast<size_t>(g_timing_repeats));
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

static std::vector<float> run_gpu_case(const std::vector<float>& input,
                                       size_t rows, size_t cols) {
  const size_t total = rows * cols;
  const size_t bytes = total * sizeof(float);
  std::vector<float> output(total, 0.0f);
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

  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, bytes));

  std::vector<float> kernel_samples;
  collect_kernel_timing_samples(d_input, d_output, bytes, rows, cols,
                                kernel_start, kernel_stop, kernel_samples);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));
  CUDA_CHECK(cudaEventElapsedTime(&g_last_timing.total_ms, total_start,
                                  total_stop));
  g_last_timing.kernel_ms = select_timing_sample(kernel_samples);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  return output;
}

static int run_tests(bool skip_cpu_verify) {
  std::vector<TestResult> results;
  bool all_ok = true;

  const std::vector<TestCase> exact_tests = {
      {"sample_4x4",
       4,
       4,
       {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, -1.5f,
        1.5f, 0.0f, -2.5f, 3.0f, -3.0f, 2.5f, -0.1f, 0.1f},
       {0.11920292f, 0.26894143f, 0.5f, 0.73105860f,
        0.88079708f, 0.37754067f, 0.62245935f, 0.18242552f,
        0.81757450f, 0.5f, 0.07585818f, 0.95257413f,
        0.04742587f, 0.92414182f, 0.47502081f, 0.52497917f}},
      {"tail_1x5",
       1,
       5,
       {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f},
       {0.01798621f, 0.26894143f, 0.5f, 0.73105860f, 0.98201376f}},
  };

  const std::vector<SizedCase> tensara_tests = {
      {"tensara_1", 4096, 4096},
      {"tensara_2", 6144, 4096},
      {"tensara_3", 4096, 7168},
      {"tensara_4", 4096, 8192},
      {"tensara_5", 8192, 8192},
  };

  const std::vector<SizedCase> medium_tests = {
      {"medium_1", 64, 64},
      {"medium_2", 255, 257},
      {"medium_rect", 512, 1025},
      {"medium_tail", 257, 258},
  };

  const std::vector<SizedCase> large_tests = {
      {"large_1", 1023, 2049},
      {"large_2", 1537, 2049},
  };

  const KernelVariant kernel_variants[] = {KernelVariant::kBasic,
                                           KernelVariant::kFloat4};

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
      cpu_sigmoid_reference(input, ref, rows, cols);
      res.cpu = "REF";
      if (expected != nullptr) {
        const bool cpu_ok = verify_close(ref, *expected, 6e-5f, 1e-4f,
                                         name.c_str(), false);
        res.cpu = cpu_ok ? "PASS" : "FAIL";
        all_ok &= cpu_ok;
      }
    }

    if (kGpuKernelImplemented) {
      const auto gpu = run_gpu_case(input, rows, cols);
      if (expected != nullptr) {
        const bool gpu_ok = verify_close(gpu, *expected, 6e-5f, 1e-4f,
                                         name.c_str(), false);
        res.gpu = gpu_ok ? "PASS" : "FAIL";
        all_ok &= gpu_ok;
      } else if (!skip_cpu_verify && kCpuReferenceImplemented) {
        std::vector<float> ref(rows * cols, 0.0f);
        cpu_sigmoid_reference(input, ref, rows, cols);
        const bool gpu_ok = verify_close(gpu, ref, 6e-5f, 1e-4f,
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

    for (const auto& tc : exact_tests) {
      record_case("small", tc.name, tc.input, &tc.expected, tc.rows, tc.cols);
    }

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      for (const auto& tc : medium_tests) {
        const std::vector<float> input = make_sigmoid_input(tc.rows, tc.cols);
        record_case("medium", tc.name, input, nullptr, tc.rows, tc.cols);
      }
      for (const auto& tc : large_tests) {
        const std::vector<float> input = make_sigmoid_input(tc.rows, tc.cols);
        record_case("large", tc.name, input, nullptr, tc.rows, tc.cols);
      }
    }

    if (skip_cpu_verify) {
      for (const auto& tc : tensara_tests) {
        const std::vector<float> input = make_sigmoid_input(tc.rows, tc.cols);
        record_case("tensara", tc.name, input, nullptr, tc.rows, tc.cols);
      }
    }
  }

  g_kernel_variant = KernelVariant::kBasic;
  std::cout << "Timing samples: mode=" << timing_mode_name()
            << " repeats=" << g_timing_repeats
            << " warmup=" << kTimingWarmupIterations
            << " metric=kernel_ms\n";
  std::cout << "CPU reference implemented: "
            << (kCpuReferenceImplemented ? "yes" : "no") << '\n';
  std::cout << "GPU kernel implemented: "
            << (kGpuKernelImplemented ? "yes" : "no") << "\n\n";

  print_results_table(results);
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
                << " (supported: --skip-cpu, --kernel=..., "
                << "--timing=..., --timing-repeats=...)\n";
      return 1;
    }
  }

  return run_tests(skip_cpu_verify);
}
