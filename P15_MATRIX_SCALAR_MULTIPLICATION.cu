#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Problem 15: Matrix Scalar Multiplication
 * Source: https://tensara.org/problems/matrix-scalar
 *
 * input_matrix and output_matrix are contiguous N by N float32 matrices.
 * Compute output_matrix[i] = input_matrix[i] * scalar for all N * N values.
 *
 * Published shapes:
 * - N=8192, scalar=0.1, 0.2, -0.3, 0.4, -0.5
 * - N=9216, scalar=0.1, 0.2, -0.3, 0.4, -0.5
 */

extern "C" void solution(const float* input_matrix, float scalar,
                         float* output_matrix, size_t N);

static constexpr bool kCpuReferenceImplemented = true;
static constexpr bool kGpuKernelImplemented = true;
static constexpr int kDefaultTimingRepeats = 5;
static constexpr int kTimingWarmupRounds = 5;
static constexpr unsigned kTimingShuffleSeed = 0x5EED15u;

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

[[maybe_unused]] static LaunchConfig g_launch_config{256, 64};
static KernelVariant g_kernel_variant = KernelVariant::kBasic;
static TimingMode g_timing_mode = TimingMode::kMedian;
static int g_timing_repeats = kDefaultTimingRepeats;

static const char* kernel_variant_name(KernelVariant variant) {
  return variant == KernelVariant::kBasic ? "basic" : "float4";
}

static const char* timing_mode_name() {
  return g_timing_mode == TimingMode::kMedian ? "median" : "best";
}

static float select_timing_sample(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  return g_timing_mode == TimingMode::kBest ? samples.front()
                                             : samples[samples.size() / 2];
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
  std::cerr << "Unknown timing mode: " << value << '\n';
  return false;
}

static bool parse_timing_repeats_arg(const std::string& arg) {
  const std::string prefix = "--timing-repeats=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  try {
    size_t end = 0;
    const int repeats = std::stoi(arg.substr(prefix.size()), &end);
    if (end != arg.size() - prefix.size() || repeats < 1) {
      return false;
    }
    g_timing_repeats = repeats;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

static bool cuda_runtime_ready() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "CUDA runtime unavailable\n";
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
            << ") at " << file << ':' << line << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__, #expr)

struct TestCase {
  const char* name = "";
  size_t N = 0;
  float scalar = 0.0f;
  std::vector<float> input;
};

// Keep this body empty until implementation is requested.
static void cpu_matrix_scalar_reference(const std::vector<float>& input,
                                        float scalar,
                                        std::vector<float>& output,
                                        size_t N) {
  for (size_t gx = 0; gx < N * N; gx ++) {
    output[gx] = input[gx] * scalar;
  }
}

// Keep this body empty until implementation is requested.
__global__ void matrix_scalar_basic_kernel(const float* input_matrix,
                                           float scalar,
                                           float* output_matrix,
                                           size_t N) {

  size_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gid; gx < N * N; gx += grid_stride)
    output_matrix[gx] = input_matrix[gx] * scalar;
}

__global__ void matrix_scalar_float4_kernel(const float* input_matrix,
                                            float scalar,
                                            float* output_matrix,
                                            size_t N) {
  const size_t total = N * N;
  const size_t total4 = total / 4;
  const size_t tail_start = total4 * 4;
  const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;

  const float4* input4 = reinterpret_cast<const float4*>(input_matrix);
  float4* output4 = reinterpret_cast<float4*>(output_matrix);

  for (size_t i = gid; i < total4; i += grid_stride) {
    const float4 value = input4[i];
    output4[i] = make_float4(value.x * scalar, value.y * scalar,
                              value.z * scalar, value.w * scalar);
  }

  for (size_t i = tail_start + gid; i < total; i += grid_stride)
    output_matrix[i] = input_matrix[i] * scalar;
}

extern "C" void solution(const float* input_matrix, float scalar,
                         float* output_matrix, size_t N) {
  const dim3 block_shape(g_launch_config.block_x, 1, 1);
  const dim3 grid_shape(g_launch_config.grid_x, 1, 1);
  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      matrix_scalar_basic_kernel<<<grid_shape, block_shape>>>(
          input_matrix, scalar, output_matrix, N);
      break;
    case KernelVariant::kFloat4:
      matrix_scalar_float4_kernel<<<grid_shape, block_shape>>>(
          input_matrix, scalar, output_matrix, N);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

static size_t matrix_element_count(size_t N) {
  return N * N;
}

static std::vector<float> make_matrix_input(size_t N) {
  const size_t count = matrix_element_count(N);
  std::vector<float> input(count);
  for (size_t i = 0; i < count; ++i) {
    const int raw = static_cast<int>((i * 31 + N * 17 + 7) % 251) - 125;
    input[i] = static_cast<float>(raw) / 37.0f;
  }
  return input;
}

struct TestResult {
  std::string group;
  std::string name;
  std::string kernel;
  size_t N = 0;
  float scalar = 0.0f;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float kernel_ms = 0.0f;
};

struct GpuJob {
  KernelVariant kernel_variant = KernelVariant::kBasic;
  LaunchConfig launch_config;
  size_t result_index = 0;
  std::vector<float> samples;
};

static void launch_gpu_job(const GpuJob& job, const float* input,
                           float* output, const TestCase& test) {
  g_launch_config = job.launch_config;
  g_kernel_variant = job.kernel_variant;
  solution(input, test.scalar, output, test.N);
}

static bool verify_close(const std::vector<float>& got,
                         const std::vector<float>& expected,
                         const char* label) {
  if (got.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-5f) {
      std::cerr << "verify(" << label << "): FAIL at index " << i << '\n';
      return false;
    }
  }
  return true;
}

static bool run_gpu_jobs(const TestCase& test, std::vector<GpuJob>& jobs,
                         std::vector<TestResult>& results,
                         const std::vector<float>* reference) {
  const size_t count = matrix_element_count(test.N);
  const size_t bytes = count * sizeof(float);
  float* input = nullptr;
  float* output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, bytes));
  CUDA_CHECK(cudaMalloc(&output, bytes));
  CUDA_CHECK(cudaMemcpy(input, test.input.data(), bytes,
                        cudaMemcpyHostToDevice));

  std::vector<size_t> order(jobs.size());
  for (size_t i = 0; i < jobs.size(); ++i) {
    order[i] = i;
  }
  std::mt19937 random(kTimingShuffleSeed ^ static_cast<unsigned>(test.N));
  for (int round = 0; round < kTimingWarmupRounds; ++round) {
    std::shuffle(order.begin(), order.end(), random);
    for (size_t index : order) {
      launch_gpu_job(jobs[index], input, output, test);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int round = 0; round < g_timing_repeats; ++round) {
    std::shuffle(order.begin(), order.end(), random);
    for (size_t index : order) {
      CUDA_CHECK(cudaEventRecord(start));
      launch_gpu_job(jobs[index], input, output, test);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      jobs[index].samples.push_back(elapsed_ms);
    }
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  for (const GpuJob& job : jobs) {
    results[job.result_index].kernel_ms = select_timing_sample(job.samples);
  }

  bool all_ok = true;
  if (reference != nullptr) {
    std::vector<float> host_output(count);
    for (const GpuJob& job : jobs) {
      launch_gpu_job(job, input, output, test);
      CUDA_CHECK(cudaMemcpy(host_output.data(), output, bytes,
                            cudaMemcpyDeviceToHost));
      const bool ok = verify_close(host_output, *reference, test.name);
      results[job.result_index].gpu = ok ? "PASS" : "FAIL";
      all_ok &= ok;
    }
  }

  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(output));
  return all_ok;
}

static void print_results(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(10) << "group" << std::setw(28)
            << "name" << std::setw(10) << "kernel" << std::setw(8)
            << "N" << std::setw(10) << "scalar"
            << std::setw(8) << "block" << std::setw(8) << "grid"
            << std::setw(8) << "cpu" << std::setw(8) << "gpu"
            << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(100, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);
  for (const TestResult& result : results) {
    std::cout << std::left << std::setw(10) << result.group
              << std::setw(28) << result.name << std::setw(10)
              << result.kernel << std::setw(8) << result.N
              << std::setw(10) << result.scalar << std::setw(8)
              << result.block_x << std::setw(8) << result.grid_x
              << std::setw(8) << result.cpu << std::setw(8) << result.gpu
              << std::setw(12) << result.kernel_ms << '\n';
  }
}

static int run_tests(bool skip_cpu_verify) {
  if (kGpuKernelImplemented && !cuda_runtime_ready()) {
    return 1;
  }
  std::cout << "Timing samples: mode=" << timing_mode_name()
            << " repeats=" << g_timing_repeats
            << " warmup_rounds=" << kTimingWarmupRounds << '\n';
  std::cout << "CPU reference implemented: "
            << (kCpuReferenceImplemented ? "yes" : "no") << '\n';
  std::cout << "GPU kernel implemented: "
            << (kGpuKernelImplemented ? "yes" : "no") << "\n\n";

  const std::vector<TestCase> exact_tests = {
      {"sample_8x8", 8, -0.5f,
       {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
        0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
        1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f,
        2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f, 3.1f,
        3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f, 3.8f, 3.9f,
        4.0f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f, 4.7f,
        4.8f, 4.9f, 5.0f, 5.1f, 5.2f, 5.3f, 5.4f, 5.5f,
        5.6f, 5.7f, 5.8f, 5.9f, 6.0f, 6.1f, 6.2f, 6.3f}},
      {"small_3x3", 3, 2.0f,
       {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}},
  };
  const size_t matrix_sizes[] = {8192, 9216};
  const float scalars[] = {0.1f, 0.2f, -0.3f, 0.4f, -0.5f};
  const LaunchConfig default_config{256, 64};
  const int block_sizes[] = {64, 128, 256, 512};
  const int grid_sizes[] = {8, 16, 32, 64, 128};

  std::vector<TestResult> results;
  bool all_ok = true;
  auto run_group = [&](const char* group, TestCase test,
                       const std::vector<LaunchConfig>& configs) {
    if (test.input.empty()) {
      test.input = make_matrix_input(test.N);
    }
    std::vector<float> reference;
    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      reference.assign(matrix_element_count(test.N), 0.0f);
      cpu_matrix_scalar_reference(test.input, test.scalar, reference, test.N);
    }
    std::vector<GpuJob> jobs;
    const KernelVariant variants[] = {
        KernelVariant::kBasic,
        KernelVariant::kFloat4,
    };
    for (const KernelVariant variant : variants) {
      for (const LaunchConfig& config : configs) {
        TestResult result;
        result.group = group;
        result.name = test.name;
        result.kernel = kernel_variant_name(variant);
        result.N = test.N;
        result.scalar = test.scalar;
        result.block_x = config.block_x;
        result.grid_x = config.grid_x;
        result.cpu = reference.empty() ? "SKIP" : "REF";
        result.gpu = "SKIP";
        results.push_back(result);
        jobs.push_back({variant, config, results.size() - 1, {}});
      }
    }
    if (kGpuKernelImplemented) {
      const std::vector<float>* target =
          reference.empty() ? nullptr : &reference;
      all_ok &= run_gpu_jobs(test, jobs, results, target);
    }
  };

  if (!skip_cpu_verify) {
    for (const TestCase& test : exact_tests) {
      run_group("exact", test, {default_config});
    }
  }
  if (skip_cpu_verify && kGpuKernelImplemented) {
    std::vector<LaunchConfig> configs;
    for (int block_x : block_sizes) {
      for (int grid_x : grid_sizes) {
        configs.push_back({block_x, grid_x});
      }
    }
    for (size_t N : matrix_sizes) {
      for (float scalar : scalars) {
        const std::string name = std::to_string(N) + "x" +
                                 std::to_string(N) + " scalar=" +
                                 std::to_string(scalar);
        run_group("sweep", {name.c_str(), N, scalar, {}}, configs);
      }
    }
  }
  print_results(results);
  return all_ok ? 0 : 1;
}

int main(int argc, char** argv) {
  bool skip_cpu_verify = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--skip-cpu") {
      skip_cpu_verify = true;
    } else if (arg.rfind("--timing=", 0) == 0) {
      if (!parse_timing_arg(arg)) {
        return 1;
      }
    } else if (arg.rfind("--timing-repeats=", 0) == 0) {
      if (!parse_timing_repeats_arg(arg)) {
        return 1;
      }
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--skip-cpu] [--timing=median|best]"
                << " [--timing-repeats=N]\n";
      return 1;
    }
  }
  return run_tests(skip_cpu_verify);
}
