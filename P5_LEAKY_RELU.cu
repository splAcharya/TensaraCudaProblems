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
// - input and output are device pointers
// - alpha is the slope for negative values
// - input/output are row-major matrices with shape (m, n)
extern "C" void solution(const float* input, float alpha, float* output,
                         size_t n, size_t m);

// CPU and GPU implementations are available for correctness verification.
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
  float alpha = 0.0f;
  std::vector<float> input;
  std::vector<float> expected;
};

// CPU reference implementation.
//
// input: row-major matrix flattened as a vector with shape (rows * cols)
// alpha: slope used for negative input values
// output: row-major matrix flattened as a vector with the same shape
// rows: number of matrix rows
// cols: number of matrix columns
static void cpu_leaky_relu(const std::vector<float>& input, float alpha,
                           std::vector<float>& output, size_t rows,
                           size_t cols) {

  const size_t total = rows * cols;
  for (size_t i = 0; i < total; ++i)
    output[i] = (input[i] >= 0.0f) ? input[i] : (alpha * input[i]);
}

static std::vector<float> make_leaky_relu_input(size_t rows, size_t cols) {
  const size_t total = rows * cols;
  std::vector<float> input(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 37 + rows * 11 + cols * 7 + 23) % 257) - 128;
    float x = static_cast<float>(raw) / 31.0f;
    if (i % 19 == 0) {
      x = 0.0f;
    } else if (i % 7 == 0) {
      x = -std::fabs(x);
    } else if (i % 5 == 0) {
      x = std::fabs(x);
    }
    input[i] = x;
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
  float alpha = 0.0f;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group" << std::setw(16) << "name"
            << std::setw(12) << "kernel" << std::setw(10) << "rows"
            << std::setw(10) << "cols" << std::setw(8) << "alpha"
            << std::setw(8) << "block_x" << std::setw(8) << "grid_x"
            << std::setw(6) << "cpu" << std::setw(6) << "gpu"
            << std::setw(12) << "total_ms"
            << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(116, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& r : results) {
    std::cout << std::left << std::setw(8) << r.group << std::setw(16)
              << r.name << std::setw(12) << r.kernel << std::setw(10)
              << r.rows << std::setw(10) << r.cols << std::setw(8)
              << r.alpha << std::setw(8) << r.block_x << std::setw(8)
              << r.grid_x << std::setw(6) << r.cpu << std::setw(6)
              << r.gpu << std::setw(12) << r.total_ms << std::setw(12)
              << r.kernel_ms << '\n';
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

static void collect_kernel_timing_samples(const float* d_input, float alpha,
                                          float* d_output, size_t bytes,
                                          size_t cols, size_t rows,
                                          cudaEvent_t kernel_start,
                                          cudaEvent_t kernel_stop,
                                          std::vector<float>& samples) {
  for (int i = 0; i < kTimingWarmupIterations; ++i) {
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));
    solution(d_input, alpha, d_output, cols, rows);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  samples.clear();
  samples.reserve(g_timing_repeats);
  for (int i = 0; i < g_timing_repeats; ++i) {
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    solution(d_input, alpha, d_output, cols, rows);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    samples.push_back(kernel_ms);
  }
}

static void run_solution_host(const std::vector<float>& input, float alpha,
                              std::vector<float>& output, size_t cols,
                              size_t rows) {
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
  collect_kernel_timing_samples(d_input, alpha, d_output, bytes, cols, rows,
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
  const float alpha = 0.1f;
  const size_t total = rows * cols;
  const size_t bytes = total * sizeof(float);
  const auto input = make_leaky_relu_input(rows, cols);

  float* d_input = nullptr;
  float* d_output = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(
      cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, bytes));

  for (int i = 0; i < kProfileWarmupIterations; ++i) {
    solution(d_input, alpha, d_output, cols, rows);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));
  CUDA_CHECK(cudaEventRecord(kernel_start));
  for (int i = 0; i < kProfileIterations; ++i) {
    solution(d_input, alpha, d_output, cols, rows);
  }
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
  const float avg_kernel_ms = kernel_ms / kProfileIterations;
  const size_t device_bytes = bytes * 2;
  const double device_mib =
      static_cast<double>(device_bytes) / (1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(3)
            << "profile scope=kernel-only verify=off"
            << " kernel=" << current_kernel_name()
            << " rows=" << rows << " cols=" << cols
            << " alpha=" << alpha
            << " block_x=" << g_launch_config.block_x
            << " grid_x=" << g_launch_config.grid_x
            << " warmup=" << kProfileWarmupIterations
            << " repeats=" << kProfileIterations
            << " device_mib=" << device_mib
            << " avg_kernel_ms=" << avg_kernel_ms << '\n';

  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return 0;
}

static int run_tests(bool skip_cpu_verify) {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const LaunchConfig default_launch = g_launch_config;
  const std::vector<TestCase> tests = {
      {"small_1",
       4,
       4,
       0.01f,
       {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, -1.5f,
        1.5f, 0.0f, -2.5f, 3.0f, -3.0f, 2.5f, -0.2f, 0.4f},
       {-0.02f, -0.01f, 0.0f, 1.0f, 2.0f, -0.005f, 0.5f, -0.015f,
        1.5f, 0.0f, -0.025f, 3.0f, -0.03f, 2.5f, -0.002f, 0.4f}},
      {"small_2",
       2,
       3,
       0.1f,
       {-4.0f, -1.0f, 2.0f, 0.0f, 3.0f, -0.5f},
       {-0.4f, -0.1f, 2.0f, 0.0f, 3.0f, -0.05f}},
      {"small_3",
       3,
       1,
       0.2f,
      {-2.5f, 0.0f, 4.0f},
      {-0.5f, 0.0f, 4.0f}},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
    float alpha;
  } medium_tests[] = {
      {"medium_1", 64, 64, 0.01f},
      {"medium_2", 255, 257, 0.05f},
      {"medium_3", 513, 1025, 0.1f},
      {"medium_4", 1024, 1024, 0.2f},
      {"medium_tail", 257, 258, 0.2f},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
    float alpha;
  } large_verify_tests[] = {
      {"large_1", 1023, 2049, 0.05f},
      {"large_2", 1537, 2049, 0.1f},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
    float alpha;
  } tensara_tests[] = {
      {"tensara_1", 4096, 4096, 0.01f},
      {"tensara_2", 4096, 4096, 0.05f},
      {"tensara_3", 4096, 4096, 0.1f},
      {"tensara_4", 4096, 4096, 0.2f},
      {"tensara_5", 6144, 4096, 0.01f},
      {"tensara_6", 6144, 4096, 0.05f},
      {"tensara_7", 6144, 4096, 0.1f},
      {"tensara_8", 6144, 4096, 0.2f},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
    float alpha;
  } shape_tests[] = {
      {"shape_rect", 2049, 4097, 0.1f},
      {"shape_wide", 1024, 8192, 0.05f},
      {"shape_tall", 8192, 1024, 0.2f},
      {"shape_tail", 4097, 8193, 0.01f},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
    float alpha;
  } tail_tests[] = {
      {"tail_1", 2049, 2049, 0.01f},
      {"tail_2", 3073, 4097, 0.05f},
      {"tail_3", 4097, 8193, 0.1f},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
    float alpha;
  } scale_tests[] = {
      {"scale_sq", 4096, 4096, 0.01f},
      {"scale_rect_1", 6144, 4096, 0.1f},
      {"scale_rect_2", 4096, 8192, 0.05f},
  };

  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};
  const KernelVariant kernel_variants[] = {KernelVariant::kBasic,
                                           KernelVariant::kFloat4};

  bool all_ok = true;
  std::vector<TestResult> results;

  auto run_sized = [&](const char* group, const char* name, size_t rows,
                       size_t cols, float alpha) {
    g_launch_config = default_launch;

    const auto input = make_leaky_relu_input(rows, cols);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref.assign(input.size(), 0.0f);
      cpu_leaky_relu(input, alpha, ref, rows, cols);
      cpu_status = "REF";
    }

    std::vector<float> gpu_out(input.size(), 0.0f);
    run_solution_host(input, alpha, gpu_out, cols, rows);

    TestResult res;
    res.group = group;
    res.name = name;
    res.kernel = current_kernel_name();
    res.rows = rows;
    res.cols = cols;
    res.alpha = alpha;
    res.block_x = g_launch_config.block_x;
    res.grid_x = g_launch_config.grid_x;
    res.cpu = cpu_status;

    if (!skip_cpu_verify && kCpuReferenceImplemented &&
        kGpuKernelImplemented) {
      const bool gpu_ok =
          verify_close(gpu_out, ref, 1e-4f, 1e-4f, name, false);
      all_ok &= gpu_ok;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
    } else {
      res.gpu = "SKIP";
    }

    res.total_ms = g_last_timing.total_ms;
    res.kernel_ms = g_last_timing.kernel_ms;
    results.push_back(res);
  };

  auto run_scaling = [&](const char* name, size_t rows, size_t cols,
                         float alpha) {
    const auto input = make_leaky_relu_input(rows, cols);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref.assign(input.size(), 0.0f);
      cpu_leaky_relu(input, alpha, ref, rows, cols);
      cpu_status = "REF";
    }

    for (int block_x : scale_block_sizes) {
      for (int grid_x : scale_grid_sizes) {
        g_launch_config = {block_x, grid_x};

        std::vector<float> gpu_out(input.size(), 0.0f);
        run_solution_host(input, alpha, gpu_out, cols, rows);

        TestResult res;
        res.group = "scale";
        res.name = name;
        res.kernel = current_kernel_name();
        res.rows = rows;
        res.cols = cols;
        res.alpha = alpha;
        res.block_x = g_launch_config.block_x;
        res.grid_x = g_launch_config.grid_x;
        res.cpu = cpu_status;

        if (!skip_cpu_verify && kCpuReferenceImplemented &&
            kGpuKernelImplemented) {
          const bool gpu_ok =
              verify_close(gpu_out, ref, 1e-4f, 1e-4f, name, false);
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
        std::vector<float> ref(tc.input.size(), 0.0f);
        cpu_leaky_relu(tc.input, tc.alpha, ref, tc.rows, tc.cols);
        const bool cpu_ok =
            verify_close(ref, tc.expected, 1e-5f, 1e-5f, tc.name, false);
        cpu_status = cpu_ok ? "PASS" : "FAIL";
        all_ok &= cpu_ok;
      }

      std::vector<float> gpu_out(tc.input.size(), 0.0f);
      run_solution_host(tc.input, tc.alpha, gpu_out, tc.cols, tc.rows);

      TestResult res;
      res.group = "small";
      res.name = tc.name;
      res.kernel = current_kernel_name();
      res.rows = tc.rows;
      res.cols = tc.cols;
      res.alpha = tc.alpha;
      res.block_x = g_launch_config.block_x;
      res.grid_x = g_launch_config.grid_x;
      res.cpu = cpu_status;

      if (kGpuKernelImplemented) {
        const bool gpu_ok =
            verify_close(gpu_out, tc.expected, 1e-5f, 1e-5f, tc.name, false);
        all_ok &= gpu_ok;
        res.gpu = gpu_ok ? "PASS" : "FAIL";
      } else {
        res.gpu = "SKIP";
      }

      res.total_ms = g_last_timing.total_ms;
      res.kernel_ms = g_last_timing.kernel_ms;
      results.push_back(res);
    }

    for (const auto& mt : medium_tests) {
      run_sized("medium", mt.name, mt.rows, mt.cols, mt.alpha);
    }

    for (const auto& lt : large_verify_tests) {
      run_sized("large", lt.name, lt.rows, lt.cols, lt.alpha);
    }

    if (!skip_cpu_verify) {
      run_scaling("scale_tail", 257, 258, 0.2f);
      run_scaling("scale_rect", 513, 1025, 0.1f);
    }

    if (skip_cpu_verify) {
      for (const auto& tt : tensara_tests) {
        run_sized("tensara", tt.name, tt.rows, tt.cols, tt.alpha);
      }
      for (const auto& st : shape_tests) {
        run_sized("shape", st.name, st.rows, st.cols, st.alpha);
      }
      for (const auto& tt : tail_tests) {
        run_sized("tail", tt.name, tt.rows, tt.cols, tt.alpha);
      }
      for (const auto& sc : scale_tests) {
        run_scaling(sc.name, sc.rows, sc.cols, sc.alpha);
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

// Basic GPU kernel implementation.
//
// input: device pointer to a row-major matrix with shape (m, n),
//        stored as input[row * n + col]
// alpha: slope used for negative input values
// output: device pointer to a row-major matrix with shape (m, n)
// total: number of matrix elements, normally m * n from solution(...)
__global__ void device_leaky_relu_basic(const float* input, float alpha,
                                        float* output, size_t total) {

  size_t gid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gid_x; gx < total; gx += grid_stride)
  {
    float val = input[gx];
    output[gx] = (val >= 0.0f) ? val : (alpha * val);
  }
}

// Float4 GPU kernel implementation.
//
// input: device pointer to a row-major matrix with shape (m, n),
//        stored as input[row * n + col]
// alpha: slope used for negative input values
// output: device pointer to a row-major matrix with shape (m, n)
// total: number of matrix elements, normally m * n from solution(...)
// note: implementation should handle any scalar tail after float4 work
__global__ void device_leaky_relu_float4(const float* input, float alpha,
                                         float* output, size_t total) {

  const float4 *input_vec = reinterpret_cast<const float4 *>(input);
  float4 *output_vec = reinterpret_cast<float4 *>(output);
  size_t total_vec = total/4;

  const size_t gix = (blockDim.x * blockIdx.x) + threadIdx.x;
  const size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gix; gx < total_vec; gx += grid_stride)
  {
    float4 ivec = input_vec[gx];
    ivec.w = (ivec.w >= 0) ? ivec.w : (alpha * ivec.w);
    ivec.x = (ivec.x >= 0) ? ivec.x : (alpha * ivec.x);
    ivec.y = (ivec.y >= 0) ? ivec.y : (alpha * ivec.y);
    ivec.z = (ivec.z >= 0) ? ivec.z : (alpha * ivec.z);
    output_vec[gx] = ivec;
  }

  //tail part
  const size_t tail_start = (total_vec * 4) + gix;
  for (size_t gx = tail_start; gx < total; gx += grid_stride)
  {
    float ival = input[gx];
    output[gx] = (ival >= 0) ? ival : (alpha * ival);
  }
}

extern "C" void solution(const float* input, float alpha, float* output,
                         size_t n, size_t m) {
  const size_t total = n * m;
  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      device_leaky_relu_basic<<<grid_shape, block_shape>>>(input, alpha,
                                                           output, total);
      break;
    case KernelVariant::kFloat4:
      device_leaky_relu_float4<<<grid_shape, block_shape>>>(input, alpha,
                                                            output, total);
      break;
  }

  CUDA_CHECK(cudaGetLastError());
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
