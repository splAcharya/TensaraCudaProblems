#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <limits>

/*
 * Problem 14: 1D Max Pooling
 * Source: https://tensara.org/problems/max-pool-1d
 *
 * For each output position i, take the maximum value in a window with
 * kernel size K, stride S, padding P, and dilation D:
 *
 * output[i] = max(input[S * i + D * m - P]), for 0 <= m < K.
 *
 * Positions outside the input range contribute negative infinity. The output
 * size is floor((H + 2P - D(K - 1) - 1) / S) + 1.
 *
 * Published shapes:
 * - H=2097152, K=7, S=4, P=3, D=1
 * - H=4194304, K=2, S=1, P=0, D=1
 * - H=8388608, K=3, S=2, P=1, D=1
 * - H=16777216, K=4, S=2, P=1, D=2
 * - H=33554432, K=3, S=1, P=1, D=1
 * - H=67108864, K=5, S=3, P=2, D=1
 */

// Tensara-style signature:
// - input and output are device pointers to float32 arrays
// - input has length H
// - output has length H_out from the formula above
extern "C" void solution(const float* input, int kernel_size, int stride,
                         int padding, int dilation, float* output, size_t H);

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
  kShared,
};

[[maybe_unused]] static LaunchConfig g_launch_config{256, 64};
static KernelVariant g_kernel_variant = KernelVariant::kBasic;
static TimingMode g_timing_mode = TimingMode::kMedian;
static int g_timing_repeats = kDefaultTimingRepeats;

static const char* kernel_variant_name(KernelVariant variant) {
  return variant == KernelVariant::kBasic ? "basic" : "shared";
}

static const char* timing_mode_name() {
  return g_timing_mode == TimingMode::kMedian ? "median" : "best";
}

static float select_timing_sample(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  if (g_timing_mode == TimingMode::kBest) {
    return samples.front();
  }
  return samples[samples.size() / 2];
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
    std::cerr << "CUDA runtime unavailable: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  if (device_count == 0) {
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
            << ") at " << file << ':' << line << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__, #expr)

static size_t max_pool_output_size(size_t H, int kernel_size, int stride,
                                   int padding, int dilation) {
  if (kernel_size <= 0 || stride <= 0 || padding < 0 || dilation <= 0) {
    return 0;
  }

  const long long numerator =
      static_cast<long long>(H) + (2LL * padding) -
      (static_cast<long long>(dilation) * (kernel_size - 1)) - 1;
  if (numerator < 0) {
    return 0;
  }

  return static_cast<size_t>((numerator / stride) + 1);
}

struct TestCase {
  const char* name = "";
  size_t H = 0;
  int kernel_size = 0;
  int stride = 0;
  int padding = 0;
  int dilation = 0;
  std::vector<float> input;
};

static void cpu_max_pool_1d_reference(const std::vector<float>& input,
                                      int kernel_size, int stride,
                                      int padding, int dilation,
                                      std::vector<float>& output, 
                                      size_t H) {
  
  int k_eff = dilation * (kernel_size - 1) + 1;
  for (int gx  = 0; gx < output.size(); ++gx)
  {
    //compute bounds and clamp
    int ws = (gx * stride) - padding;
    int we = ws + k_eff;

    //read input
    float maxv = std::numeric_limits<float>::lowest();
    for (int i = ws; i < we; i += dilation)
    {
      if (0 <= i && i < (int)H)
        maxv = std::max(maxv, input[i]);
    }
  
    output[gx] = maxv;
  }
}

__global__ void max_pool_1d_basic_kernel(const float* input, int kernel_size,
                                         int stride, int padding,
                                         int dilation, float* output,
                                         size_t H, size_t H_out) {

  size_t gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gid; gx < H_out; gx += grid_stride)
  {
    //read bounds
    int k_eff = dilation * (kernel_size - 1) + 1;
    int ws = ((int)gx * stride) - padding;
    int we = ws + k_eff;

    //read gmem
    float maxv = -FLT_MAX;
    for (int i = ws; i < we; i += dilation)
    {
      if ( 0 <= i && i < H)
      {
        const float value = input[i];
        if (value > maxv)
          maxv = value;
      }
    }

    output[gx] = maxv;
  }
}

__global__ void max_pool_1d_shared_kernel(const float* input, int kernel_size,
                                          int stride, int padding,
                                          int dilation, float* output,
                                          size_t H, size_t H_out) {
  
  size_t total_blocks = (H_out + blockDim.x - 1) / blockDim.x;
  extern __shared__ float smem_ar[];

  for (size_t bx = blockIdx.x; bx < total_blocks; bx += gridDim.x)
  {
    int k_eff       = dilation * (kernel_size - 1) + 1;
    int output_ws   = (bx * blockDim.x);
    int input_ws    = (output_ws * stride) - padding;
    int output_we   = output_ws + blockDim.x - 1;
    int input_we    = ((output_we * stride) - padding) + k_eff;
    int total_elems = input_we - input_ws;
    
    //cooperatively load from gmem to smem
    for (int lx = threadIdx.x; lx < total_elems; lx += blockDim.x)
    {
      int load_idx = input_ws + lx;
      if (0 <= load_idx && load_idx < H)
        smem_ar[lx] = input[load_idx];
      else
        smem_ar[lx] = -FLT_MAX; 
    }

    __syncthreads();

    //local pooling
    int ws = (threadIdx.x * stride);
    int we = ws + k_eff;

    float maxv = -FLT_MAX; 
    for (int i = ws; i < we; i += dilation)
    {
      if ( 0 <= i && i < total_elems)
      {
        const float temp = smem_ar[i];
        if (temp > maxv)
          maxv = temp;
      }
    }

    __syncthreads();
    
    size_t gx = (bx * blockDim.x) + threadIdx.x;
    if (gx < H_out)
      output[gx] = maxv;
  }
}

extern "C" void solution(const float* input, int kernel_size, int stride,
                         int padding, int dilation, float* output, size_t H) {
  const size_t H_out = max_pool_output_size(
      H, kernel_size, stride, padding, dilation);
  const int k_eff = dilation * (kernel_size - 1) + 1;
  const dim3 block_shape(g_launch_config.block_x, 1, 1);
  const dim3 grid_shape(g_launch_config.grid_x, 1, 1);
  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      max_pool_1d_basic_kernel<<<grid_shape, block_shape>>>(
          input, kernel_size, stride, padding, dilation, output, H, H_out);
      break;
    case KernelVariant::kShared: {
      const size_t total_elements =
          static_cast<size_t>(g_launch_config.block_x - 1) * stride +
          k_eff;
      const size_t shared_bytes = total_elements * sizeof(float);
      max_pool_1d_shared_kernel<<<grid_shape, block_shape, shared_bytes>>>(
          input, kernel_size, stride, padding, dilation, output, H, H_out);
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_max_pool_input(size_t H) {
  std::vector<float> input(H, 0.0f);
  for (size_t i = 0; i < H; ++i) {
    const int raw = static_cast<int>((i * 31 + H * 17 + 7) % 251) - 125;
    input[i] = static_cast<float>(raw) / 37.0f;
  }
  return input;
}

struct TestResult {
  std::string group;
  std::string name;
  std::string kernel;
  size_t H = 0;
  size_t H_out = 0;
  int kernel_size = 0;
  int stride = 0;
  int padding = 0;
  int dilation = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

struct GpuJob {
  KernelVariant kernel_variant = KernelVariant::kBasic;
  LaunchConfig launch_config;
  size_t result_index = 0;
  std::vector<float> samples;
};

static unsigned timing_seed(const TestCase& test, size_t job_count) {
  return kTimingShuffleSeed ^ static_cast<unsigned>(test.H) ^
         (static_cast<unsigned>(test.kernel_size) * 0x9E3779B9u) ^
         (static_cast<unsigned>(test.stride) * 0x85EBCA6Bu) ^
         (static_cast<unsigned>(test.padding) * 0xC2B2AE35u) ^
         (static_cast<unsigned>(test.dilation) * 0x27D4EB2Fu) ^
         static_cast<unsigned>(job_count);
}

static void launch_gpu_job(const GpuJob& job, const float* device_input,
                           float* device_output, const TestCase& test) {
  g_launch_config = job.launch_config;
  g_kernel_variant = job.kernel_variant;
  solution(device_input, test.kernel_size, test.stride, test.padding,
           test.dilation, device_output, test.H);
}

static bool verify_equal(const std::vector<float>& got,
                         const std::vector<float>& expected,
                         const char* label) {
  if (got == expected) {
    return true;
  }
  std::cerr << "verify(" << label << "): FAIL\n";
  return false;
}

static bool run_gpu_jobs(const TestCase& test,
                         const std::vector<float>& input,
                         std::vector<GpuJob>& jobs,
                         std::vector<TestResult>& results,
                         const std::vector<float>* reference) {
  const size_t input_bytes = input.size() * sizeof(float);
  const size_t output_size = max_pool_output_size(
      test.H, test.kernel_size, test.stride, test.padding, test.dilation);
  const size_t output_bytes = output_size * sizeof(float);
  float* device_input = nullptr;
  float* device_output = nullptr;
  std::vector<size_t> order;
  order.reserve(jobs.size());
  for (size_t i = 0; i < jobs.size(); ++i) {
    order.push_back(i);
  }

  CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&device_output, output_bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes,
                        cudaMemcpyHostToDevice));

  std::mt19937 random(timing_seed(test, jobs.size()));
  for (int round = 0; round < kTimingWarmupRounds; ++round) {
    std::shuffle(order.begin(), order.end(), random);
    for (size_t index : order) {
      launch_gpu_job(jobs[index], device_input, device_output, test);
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
      launch_gpu_job(jobs[index], device_input, device_output, test);
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
    TestResult& result = results[job.result_index];
    result.total_ms = select_timing_sample(job.samples);
    result.kernel_ms = result.total_ms;
  }

  bool all_ok = true;
  if (reference != nullptr) {
    std::vector<float> output(output_size, 0.0f);
    for (const GpuJob& job : jobs) {
      launch_gpu_job(job, device_input, device_output, test);
      CUDA_CHECK(cudaMemcpy(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost));
      const bool ok = verify_equal(output, *reference, test.name);
      results[job.result_index].gpu = ok ? "PASS" : "FAIL";
      all_ok &= ok;
    }
  }

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return all_ok;
}

static void print_results(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(10) << "group" << std::setw(28)
            << "name" << std::setw(10) << "kernel" << std::setw(12)
            << "H" << std::setw(12)
            << "H_out" << std::setw(8) << "K" << std::setw(8)
            << "stride" << std::setw(8) << "pad" << std::setw(8)
            << "dil" << std::setw(8) << "block" << std::setw(8)
            << "grid" << std::setw(8) << "cpu" << std::setw(8)
            << "gpu" << std::setw(12) << "total_ms" << std::setw(12)
            << "kernel_ms" << '\n';
  std::cout << std::string(160, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);
  for (const TestResult& result : results) {
    std::cout << std::left << std::setw(10) << result.group
              << std::setw(28) << result.name << std::setw(10)
              << result.kernel << std::setw(12) << result.H
              << std::setw(12) << result.H_out << std::setw(8)
              << result.kernel_size << std::setw(8) << result.stride
              << std::setw(8) << result.padding << std::setw(8)
              << result.dilation << std::setw(8) << result.block_x
              << std::setw(8) << result.grid_x << std::setw(8)
              << result.cpu << std::setw(8) << result.gpu << std::setw(12)
              << result.total_ms << std::setw(12) << result.kernel_ms << '\n';
  }
}

static int run_tests(bool skip_cpu_verify) {
  if (kGpuKernelImplemented && !cuda_runtime_ready()) {
    return 1;
  }

  std::cout << "Timing samples: mode=" << timing_mode_name()
            << " repeats=" << g_timing_repeats
            << " warmup_rounds=" << kTimingWarmupRounds
            << " shuffle_seed=" << kTimingShuffleSeed
            << " metric=kernel_ms\n";
  std::cout << "CPU reference implemented: "
            << (kCpuReferenceImplemented ? "yes" : "no") << '\n';
  std::cout << "GPU kernel implemented: "
            << (kGpuKernelImplemented ? "yes" : "no") << "\n\n";

  const std::vector<TestCase> exact_tests = {
      {"small_no_padding", 8, 2, 2, 0, 1,
       {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f, 7.0f, -8.0f}},
      {"small_padding", 5, 3, 1, 1, 1,
       {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f}},
      {"small_dilation", 7, 3, 1, 2, 2,
       {1.0f, -2.0f, 3.0f, 8.0f, -5.0f, 6.0f, 4.0f}},
  };
  const std::vector<TestCase> published_tests = {
      {"tensara_1", 2097152, 7, 4, 3, 1, {}},
      {"tensara_2", 4194304, 2, 1, 0, 1, {}},
      {"tensara_3", 8388608, 3, 2, 1, 1, {}},
      {"tensara_4", 16777216, 4, 2, 1, 2, {}},
      {"tensara_5", 33554432, 3, 1, 1, 1, {}},
      {"tensara_6", 67108864, 5, 3, 2, 1, {}},
  };
  const std::vector<TestCase> overlap_tests = {
      // P6-inspired larger cases.
      {"p6_large_1", 1048576, 7, 4, 3, 1, {}},
      {"p6_large_2", 1048583, 5, 3, 2, 1, {}},
      // High-reuse cases intended to expose shared-memory benefits.
      {"overlap_k15_s1", 16777216, 15, 1, 7, 1, {}},
      {"overlap_k15_d2_s2", 16777216, 15, 2, 14, 2, {}},
      {"overlap_k31_s1", 8388608, 31, 1, 15, 1, {}},
  };
  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};
  std::vector<LaunchConfig> sweep_configs;
  for (int block_x : scale_block_sizes) {
    for (int grid_x : scale_grid_sizes) {
      sweep_configs.push_back({block_x, grid_x});
    }
  }

  std::vector<TestResult> results;
  bool all_ok = true;
  auto run_group = [&](const char* group, const TestCase& test,
                       std::vector<float> input,
                       const std::vector<LaunchConfig>& configs) {
    const size_t output_size = max_pool_output_size(
        test.H, test.kernel_size, test.stride, test.padding, test.dilation);
    std::vector<float> reference;
    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      reference.assign(output_size, 0.0f);
      cpu_max_pool_1d_reference(input, test.kernel_size, test.stride,
                                test.padding, test.dilation, reference,
                                test.H);
    }

    std::vector<GpuJob> jobs;
    const KernelVariant variants[] = {
        KernelVariant::kBasic,
        KernelVariant::kShared,
    };
    for (const KernelVariant variant : variants) {
      for (const LaunchConfig& config : configs) {
        TestResult result;
        result.group = group;
        result.name = test.name;
        result.kernel = kernel_variant_name(variant);
        result.H = test.H;
        result.H_out = output_size;
        result.kernel_size = test.kernel_size;
        result.stride = test.stride;
        result.padding = test.padding;
        result.dilation = test.dilation;
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
      all_ok &= run_gpu_jobs(test, input, jobs, results, target);
    }
  };

  const std::vector<LaunchConfig> default_configs = {{256, 64}};
  if (!skip_cpu_verify) {
    for (const TestCase& test : exact_tests) {
      run_group("exact", test, test.input, default_configs);
    }
  }

  if (skip_cpu_verify && kGpuKernelImplemented) {
    for (const TestCase& test : published_tests) {
      run_group("sweep", test, make_max_pool_input(test.H), sweep_configs);
    }
    for (const TestCase& test : overlap_tests) {
      run_group("overlap", test, make_max_pool_input(test.H),
                sweep_configs);
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
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [--skip-cpu] [--timing=median|best]"
                << " [--timing-repeats=N]\n";
      return 0;
    } else if (arg.rfind("--timing=", 0) == 0) {
      if (!parse_timing_arg(arg)) {
        return 1;
      }
    } else if (arg.rfind("--timing-repeats=", 0) == 0) {
      if (!parse_timing_repeats_arg(arg)) {
        return 1;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      return 1;
    }
  }
  return run_tests(skip_cpu_verify);
}
