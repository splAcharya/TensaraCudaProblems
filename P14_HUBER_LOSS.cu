#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

/*
 * Problem 14: Huber Loss
 * Source: https://tensara.org/problems/huber-loss
 *
 * Compute element-wise Smooth L1 Loss (Huber Loss with delta = 1):
 *
 * diff = predictions[i] - targets[i]
 * output[i] = 0.5 * diff * diff, when abs(diff) < 1
 * output[i] = abs(diff) - 0.5, otherwise
 *
 * Published sizes: N=1048576, 4194304, 16777216, 67108864.
 */

extern "C" void solution(const float* predictions, const float* targets,
                         float* output, size_t N);

static constexpr bool kCpuReferenceImplemented = true;
static constexpr bool kGpuKernelImplemented = true;
static constexpr int kDefaultTimingRepeats = 5;
static constexpr int kTimingWarmupRounds = 5;
static constexpr unsigned kTimingShuffleSeed = 0x5EED14u;

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
  std::vector<float> predictions;
  std::vector<float> targets;
};

// Keep this body empty until implementation is requested.
static void cpu_huber_loss_reference(const std::vector<float>& predictions,
                                     const std::vector<float>& targets,
                                     std::vector<float>& output, size_t N) 
{
  for (size_t gx = 0; gx < N; gx++)
  {
    float d = predictions[gx] - targets[gx];
    float d_abs = std::abs(d);
    if (d_abs < 1.0f)
      output[gx] = 0.5f * d * d;
    else
      output[gx] = d_abs - 0.5f;
  }
}

__device__ float huber_loss_core(float p, float t)
{
  float d = p - t;
  float d_abs = abs(d);
  float res = (d_abs < 1.0f) ? (0.5f * d * d) : (d_abs - 0.5f);
  return res;
}

// Keep this body empty until implementation is requested.
__global__ void huber_loss_basic_kernel(const float* predictions,
                                        const float* targets, float* output,
                                        size_t N) {
  size_t gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gid; gx < N; gx += grid_stride)
    output[gx] = huber_loss_core(predictions[gx], targets[gx]);
}

__global__ void huber_loss_float4_kernel(const float* predictions,
                                         const float* targets, float* output,
                                         size_t N) {
  size_t N4 = N / 4;
  size_t tail_start = N4 * 4;
  size_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);
  const float4 *pred4 = reinterpret_cast<const float4 *>(predictions);
  const float4 *targ4 = reinterpret_cast<const float4 *>(targets);
  float4       *output4 = reinterpret_cast<float4 *>(output);

  //f4 portions
  for (size_t gx = gid; gx < N4; gx += grid_stride)
  {
    const float4 temp_p4 = pred4[gx];
    const float4 temp_t4 = targ4[gx];
    output4[gx] = make_float4(
                    huber_loss_core(temp_p4.x, temp_t4.x),
                    huber_loss_core(temp_p4.y, temp_t4.y),
                    huber_loss_core(temp_p4.z, temp_t4.z),
                    huber_loss_core(temp_p4.w, temp_t4.w));
  }

  //tail part
  for (size_t gx = tail_start + gid; gx < N; gx += grid_stride)
    output[gx] = huber_loss_core(predictions[gx], targets[gx]);
}

extern "C" void solution(const float* predictions, const float* targets,
                         float* output, size_t N) {
  const dim3 block_shape(g_launch_config.block_x, 1, 1);
  const dim3 grid_shape(g_launch_config.grid_x, 1, 1);
  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      huber_loss_basic_kernel<<<grid_shape, block_shape>>>(
          predictions, targets, output, N);
      break;
    case KernelVariant::kFloat4:
      huber_loss_float4_kernel<<<grid_shape, block_shape>>>(
          predictions, targets, output, N);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_huber_input(size_t N, int seed) {
  std::vector<float> input(N);
  for (size_t i = 0; i < N; ++i) {
    const int raw = static_cast<int>((i * 31 + seed * 17 + 7) % 251) - 125;
    input[i] = static_cast<float>(raw) / 37.0f;
  }
  return input;
}

struct TestResult {
  std::string group;
  std::string name;
  std::string kernel;
  size_t N = 0;
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

static void launch_gpu_job(const GpuJob& job, const float* predictions,
                           const float* targets, float* output,
                           const TestCase& test) {
  g_launch_config = job.launch_config;
  g_kernel_variant = job.kernel_variant;
  solution(predictions, targets, output, test.N);
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
  const size_t bytes = test.N * sizeof(float);
  float* predictions = nullptr;
  float* targets = nullptr;
  float* output = nullptr;
  CUDA_CHECK(cudaMalloc(&predictions, bytes));
  CUDA_CHECK(cudaMalloc(&targets, bytes));
  CUDA_CHECK(cudaMalloc(&output, bytes));
  CUDA_CHECK(cudaMemcpy(predictions, test.predictions.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(targets, test.targets.data(), bytes,
                        cudaMemcpyHostToDevice));

  std::vector<size_t> order(jobs.size());
  for (size_t i = 0; i < jobs.size(); ++i) {
    order[i] = i;
  }
  std::mt19937 random(kTimingShuffleSeed ^ static_cast<unsigned>(test.N));
  for (int round = 0; round < kTimingWarmupRounds; ++round) {
    std::shuffle(order.begin(), order.end(), random);
    for (size_t index : order) {
      launch_gpu_job(jobs[index], predictions, targets, output, test);
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
      launch_gpu_job(jobs[index], predictions, targets, output, test);
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
    std::vector<float> host_output(test.N);
    for (const GpuJob& job : jobs) {
      launch_gpu_job(job, predictions, targets, output, test);
      CUDA_CHECK(cudaMemcpy(host_output.data(), output, bytes,
                            cudaMemcpyDeviceToHost));
      const bool ok = verify_close(host_output, *reference, test.name);
      results[job.result_index].gpu = ok ? "PASS" : "FAIL";
      all_ok &= ok;
    }
  }

  CUDA_CHECK(cudaFree(predictions));
  CUDA_CHECK(cudaFree(targets));
  CUDA_CHECK(cudaFree(output));
  return all_ok;
}

static void print_results(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(10) << "group" << std::setw(24)
            << "name" << std::setw(10) << "kernel" << std::setw(14)
            << "N" << std::setw(8) << "block"
            << std::setw(8) << "grid" << std::setw(8) << "cpu"
            << std::setw(8) << "gpu" << std::setw(12) << "kernel_ms"
            << '\n';
  std::cout << std::string(92, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);
  for (const TestResult& result : results) {
    std::cout << std::left << std::setw(10) << result.group
              << std::setw(24) << result.name << std::setw(10)
              << result.kernel << std::setw(14) << result.N
              << std::setw(8) << result.block_x << std::setw(8)
              << result.grid_x << std::setw(8) << result.cpu << std::setw(8)
              << result.gpu << std::setw(12) << result.kernel_ms << '\n';
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
      {"sample", 8,
       {0.2f, 0.8f, 1.0f, 1.5f, -0.3f, -0.9f, -1.2f, 2.0f},
       {0.0f, 0.5f, 0.3f, 2.0f, 0.1f, -0.2f, -2.0f, 0.5f}},
      {"threshold_values", 6,
       {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f},
       {0.0f, 0.5f, 0.0f, -0.5f, 0.0f, 0.0f}},
  };
  const std::vector<TestCase> published_tests = {
      {"N=1048576", 1048576, {}, {}},
      {"N=4194304", 4194304, {}, {}},
      {"N=16777216", 16777216, {}, {}},
      {"N=67108864", 67108864, {}, {}},
  };
  const LaunchConfig default_config{256, 64};
  const int block_sizes[] = {64, 128, 256, 512};
  const int grid_sizes[] = {8, 16, 32, 64, 128};

  std::vector<TestResult> results;
  bool all_ok = true;
  auto run_group = [&](const char* group, TestCase test,
                       const std::vector<LaunchConfig>& configs) {
    if (test.predictions.empty()) {
      test.predictions = make_huber_input(test.N, 11);
      test.targets = make_huber_input(test.N, 23);
    }
    std::vector<float> reference;
    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      reference.assign(test.N, 0.0f);
      cpu_huber_loss_reference(test.predictions, test.targets, reference,
                               test.N);
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
    for (const TestCase& test : published_tests) {
      run_group("sweep", test, configs);
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
