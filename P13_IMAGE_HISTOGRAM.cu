#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Problem 13: Image Histogram
 * Source: https://tensara.org/problems/histogram
 *
 * Count the occurrences of every grayscale pixel intensity in a row-major
 * image. Input values are in [0, num_bins - 1], and the histogram has
 * num_bins entries.
 *
 * Published shapes:
 * - 2560 x 1440, bins=64, 128, 256
 * - 2048 x 2048, bins=64, 128, 256
 * - 4096 x 4096, bins=64, 128, 256
 */

// Tensara-style signature:
// - input and histogram are device pointers
// - height and width describe the row-major image
extern "C" void solution(const int* input, int* histogram, size_t height,
                         size_t width, size_t num_bins);

static constexpr bool kCpuReferenceImplemented = true;
static constexpr bool kGpuKernelImplemented = true;
static constexpr int kDefaultTimingRepeats = 5;
static constexpr int kTimingWarmupRounds = 5;
static constexpr unsigned kTimingShuffleSeed = 0x5EED13u;

struct LaunchConfig {
  int block_x = 256;
  int grid_x = 64;
};

enum class KernelVariant {
  kBasic,
  kShared,
  kPersistentShared,
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

static const char* kernel_name(KernelVariant variant) {
  switch (variant) {
    case KernelVariant::kBasic:
      return "basic";
    case KernelVariant::kShared:
      return "shared";
    case KernelVariant::kPersistentShared:
      return "persistent_shared";
  }
  return "unknown";
}

static bool kernel_enabled(KernelVariant variant) {
  return !g_kernel_arg_set || g_kernel_variant == variant;
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
  if (value == "shared") {
    g_kernel_variant = KernelVariant::kShared;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "persistent_shared") {
    g_kernel_variant = KernelVariant::kPersistentShared;
    g_kernel_arg_set = true;
    return true;
  }

  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic, --kernel=shared, or "
            << "--kernel=persistent_shared)\n";
  return false;
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

struct TestCase {
  const char* name = "";
  size_t height = 0;
  size_t width = 0;
  size_t num_bins = 0;
  std::vector<int> input;
};

// CPU reference implementation used by the correctness harness.
static void cpu_histogram_reference(const std::vector<int>& input,
                                    std::vector<int>& histogram,
                                    size_t height, size_t width,
                                    size_t num_bins) {

  std::fill(histogram.begin(), histogram.end(), 0);
  for (size_t i  = 0; i < height * width; ++i)
    histogram[input[i]]++;
}

__global__ void histogram_basic_kernel(const int* input, int* histogram,
                                       size_t height, size_t width,
                                       size_t num_bins) {

  size_t gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);
  size_t total = height * width;

  for (size_t gx = gid; gx < total; gx += grid_stride)
    atomicAdd(&histogram[input[gx]], 1);
}

// Keep this shared-memory kernel body empty until it is implemented.
__global__ void histogram_shared_kernel(const int* input, int* histogram,
                                        size_t height, size_t width,
                                        size_t num_bins) {
  extern __shared__ int lc_histogram[];

  size_t total = height * width;
  size_t total_blocks = (total + blockDim.x - 1) / blockDim.x;

  for (size_t bx = blockIdx.x; bx < total_blocks; bx += gridDim.x)
  {
    for (size_t lx = threadIdx.x; lx < num_bins; lx += blockDim.x)
      lc_histogram[lx] = 0.0f;

    //wait for clear to finish
    __syncthreads();

    //read from gmem
    size_t gid = (bx * blockDim.x) + threadIdx.x;
    if (gid < total)
      atomicAdd(&lc_histogram[input[gid]], 1);

    //wait for local histogrma to be update by entire block
    __syncthreads();

    //write into gmem
    for (size_t lx = threadIdx.x; lx < num_bins; lx += blockDim.x)
      atomicAdd(&histogram[lx], lc_histogram[lx]);

    __syncthreads();
  }
}

// Persistent shared histogram for grid-stride accumulation.
__global__ void histogram_persistent_shared_kernel(
    const int* input, int* histogram, size_t height, size_t width,
    size_t num_bins) {

  extern __shared__ int lc_histogram[];
  size_t total = height * width;
  size_t total_blocks = (total + blockDim.x - 1) / blockDim.x;

  //clear shared meory
  for (size_t lx = threadIdx.x; lx < num_bins; lx += blockDim.x)
    lc_histogram[lx] = 0.0f;
  
  __syncthreads();

  for (size_t bx = blockIdx.x; bx < total_blocks; bx += gridDim.x)
  {
    //read from gmem
    size_t gid = (bx * blockDim.x) + threadIdx.x;
    if (gid < total)
      atomicAdd(&lc_histogram[input[gid]], 1);

    //wait for local histogram update
    __syncthreads();
  }

  for (size_t lx = threadIdx.x; lx < num_bins; lx += blockDim.x)
    atomicAdd(&histogram[lx], lc_histogram[lx]);
}

extern "C" void solution(const int* input, int* histogram, size_t height,
                         size_t width, size_t num_bins) {
  const dim3 block_shape(g_launch_config.block_x, 1, 1);
  const dim3 grid_shape(g_launch_config.grid_x, 1, 1);
  CUDA_CHECK(cudaMemset(histogram, 0, num_bins * sizeof(int)));
  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      histogram_basic_kernel<<<grid_shape, block_shape>>>(
          input, histogram, height, width, num_bins);
      break;
    case KernelVariant::kShared: {
      const size_t shared_bytes = num_bins * sizeof(int);
      histogram_shared_kernel<<<grid_shape, block_shape, shared_bytes>>>(
          input, histogram, height, width, num_bins);
      break;
    }
    case KernelVariant::kPersistentShared: {
      const size_t shared_bytes = num_bins * sizeof(int);
      histogram_persistent_shared_kernel<<<grid_shape, block_shape,
                                           shared_bytes>>>(
          input, histogram, height, width, num_bins);
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

static std::vector<int> make_histogram_input(size_t height, size_t width,
                                             size_t num_bins) {
  const size_t size = height * width;
  std::vector<int> input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<int>((i * 37 + height * 11 + width * 19) %
                                num_bins);
  }
  return input;
}

struct TestResult {
  std::string group;
  std::string name;
  std::string kernel;
  size_t height = 0;
  size_t width = 0;
  size_t num_bins = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

struct GpuJob {
  KernelVariant variant;
  LaunchConfig launch_config;
  size_t result_index = 0;
  std::vector<float> samples;
};

static unsigned timing_seed(const TestCase& test, size_t job_count) {
  return kTimingShuffleSeed ^ static_cast<unsigned>(test.height) ^
         static_cast<unsigned>(test.width) ^
         static_cast<unsigned>(test.num_bins) ^
         static_cast<unsigned>(job_count * 0x9E3779B9u);
}

static void launch_gpu_job(const GpuJob& job, const int* device_input,
                           int* device_histogram, const TestCase& test) {
  g_kernel_variant = job.variant;
  g_launch_config = job.launch_config;
  solution(device_input, device_histogram, test.height, test.width,
           test.num_bins);
}

static bool verify_equal(const std::vector<int>& got,
                         const std::vector<int>& expected,
                         const char* label) {
  if (got == expected) {
    return true;
  }
  std::cerr << "verify(" << label << "): FAIL\n";
  return false;
}

static bool run_gpu_jobs(const TestCase& test, const std::vector<int>& input,
                         std::vector<GpuJob>& jobs,
                         std::vector<TestResult>& results,
                         const std::vector<int>* reference) {
  const size_t input_bytes = input.size() * sizeof(int);
  const size_t histogram_bytes = test.num_bins * sizeof(int);
  int* device_input = nullptr;
  int* device_histogram = nullptr;
  std::vector<size_t> order;
  order.reserve(jobs.size());
  for (size_t i = 0; i < jobs.size(); ++i) {
    order.push_back(i);
  }

  CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&device_histogram, histogram_bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes,
                        cudaMemcpyHostToDevice));

  std::mt19937 random(timing_seed(test, jobs.size()));
  for (int round = 0; round < kTimingWarmupRounds; ++round) {
    std::shuffle(order.begin(), order.end(), random);
    for (size_t index : order) {
      CUDA_CHECK(cudaMemset(device_histogram, 0, histogram_bytes));
      launch_gpu_job(jobs[index], device_input, device_histogram, test);
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
      CUDA_CHECK(cudaMemset(device_histogram, 0, histogram_bytes));
      CUDA_CHECK(cudaEventRecord(start));
      launch_gpu_job(jobs[index], device_input, device_histogram, test);
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
    std::vector<int> output(test.num_bins, 0);
    for (const GpuJob& job : jobs) {
      CUDA_CHECK(cudaMemset(device_histogram, 0, histogram_bytes));
      launch_gpu_job(job, device_input, device_histogram, test);
      CUDA_CHECK(cudaMemcpy(output.data(), device_histogram, histogram_bytes,
                            cudaMemcpyDeviceToHost));
      const bool ok = verify_equal(output, *reference, test.name);
      results[job.result_index].gpu = ok ? "PASS" : "FAIL";
      all_ok &= ok;
    }
  }

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_histogram));
  return all_ok;
}

static void print_results(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(10) << "group" << std::setw(30)
            << "name" << std::setw(20) << "kernel" << std::setw(8)
            << "height" << std::setw(8)
            << "width" << std::setw(10) << "bins" << std::setw(8)
            << "block_x" << std::setw(8) << "grid_x" << std::setw(8)
            << "cpu" << std::setw(8) << "gpu" << std::setw(12)
            << "total_ms" << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(120, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);
  for (const auto& result : results) {
    std::cout << std::left << std::setw(10) << result.group
              << std::setw(30) << result.name << std::setw(20)
              << result.kernel << std::setw(8)
              << result.height << std::setw(8) << result.width
              << std::setw(10) << result.num_bins << std::setw(8)
              << result.block_x << std::setw(8) << result.grid_x
              << std::setw(8) << result.cpu << std::setw(8) << result.gpu
              << std::setw(12) << result.total_ms << std::setw(12)
              << result.kernel_ms << '\n';
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
      {"small_4x4_b4", 4, 4, 4, {0, 1, 2, 3, 0, 1, 2, 3,
                                    3, 2, 1, 0, 3, 2, 1, 0}},
      {"small_3x5_b8", 3, 5, 8, {0, 7, 1, 6, 2, 5, 3, 4,
                                    0, 7, 1, 6, 2, 5, 3}},
  };
  const std::vector<TestCase> published_tests = {
      {"tensara_2560x1440_b64", 2560, 1440, 64, {}},
      {"tensara_2560x1440_b128", 2560, 1440, 128, {}},
      {"tensara_2560x1440_b256", 2560, 1440, 256, {}},
      {"tensara_2048x2048_b64", 2048, 2048, 64, {}},
      {"tensara_2048x2048_b128", 2048, 2048, 128, {}},
      {"tensara_2048x2048_b256", 2048, 2048, 256, {}},
      {"tensara_4096x4096_b64", 4096, 4096, 64, {}},
      {"tensara_4096x4096_b128", 4096, 4096, 128, {}},
      {"tensara_4096x4096_b256", 4096, 4096, 256, {}},
  };
  const std::vector<TestCase> medium_tests = {
      {"medium_257x263_b17", 257, 263, 17, {}},
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
  const KernelVariant kernel_variants[] = {KernelVariant::kBasic,
                                           KernelVariant::kShared,
                                           KernelVariant::kPersistentShared};
  std::vector<KernelVariant> enabled_variants;
  for (KernelVariant variant : kernel_variants) {
    if (kernel_enabled(variant)) {
      enabled_variants.push_back(variant);
    }
  }

  auto run_group = [&](const char* group, const TestCase& test,
                       std::vector<int> input,
                       const std::vector<LaunchConfig>& configs) {
    std::vector<int> reference;
    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      reference.assign(test.num_bins, 0);
      cpu_histogram_reference(input, reference, test.height, test.width,
                              test.num_bins);
    }

    std::vector<GpuJob> jobs;
    for (KernelVariant variant : enabled_variants) {
      for (const LaunchConfig& config : configs) {
        TestResult result;
        result.group = group;
        result.name = test.name;
        result.kernel = kernel_name(variant);
        result.height = test.height;
        result.width = test.width;
        result.num_bins = test.num_bins;
        result.block_x = config.block_x;
        result.grid_x = config.grid_x;
        result.cpu = reference.empty() ? "SKIP" : "REF";
        result.gpu = "SKIP";
        results.push_back(result);
        jobs.push_back({variant, config, results.size() - 1, {}});
      }
    }

    const std::vector<int>* target = reference.empty() ? nullptr : &reference;
    if (kGpuKernelImplemented) {
      all_ok &= run_gpu_jobs(test, input, jobs, results, target);
    }
  };

  const std::vector<LaunchConfig> default_configs = {{256, 64}};
  if (!skip_cpu_verify) {
    for (const auto& test : exact_tests) {
      run_group("exact", test, test.input, default_configs);
    }
    for (const auto& test : medium_tests) {
      run_group("medium", test,
                make_histogram_input(test.height, test.width, test.num_bins),
                default_configs);
    }
  }

  if (skip_cpu_verify) {
    for (const auto& test : published_tests) {
      run_group("sweep", test,
                make_histogram_input(test.height, test.width, test.num_bins),
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
                << " [--skip-cpu]"
                << " [--kernel=basic|shared|persistent_shared]"
                << " [--timing=median|best]"
                << " [--timing-repeats=N]\n";
      return 0;
    } else if (arg == "--list-kernels") {
      std::cout << "basic\nshared\npersistent_shared\n";
      return 0;
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
      std::cerr << "Unknown argument: " << arg << '\n';
      return 1;
    }
  }
  return run_tests(skip_cpu_verify);
}
