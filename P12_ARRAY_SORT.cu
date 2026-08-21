#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Problem 12: Array Sorting
 * Source: https://tensara.org/problems/array-sort
 *
 * Sort an integer array in ascending order. The input and output arrays are
 * one-dimensional and contain the same number of elements.
 *
 * Published sizes:
 * - 16384
 * - 32768
 * - 65536
 * - 131072
 * - 262144
 */

extern "C" void solution(int* input, int* output, size_t size);

static constexpr bool kCpuReferenceImplemented = true;
static constexpr bool kGpuKernelImplemented = true;
static constexpr int kDefaultTimingRepeats = 5;
static constexpr int kTimingWarmupRounds = 5;
static constexpr unsigned kTimingShuffleSeed = 0x5EED12u;

struct LaunchConfig {
  int block_x = 256;
  int grid_x = 64;
};

enum class TimingMode {
  kMedian,
  kBest,
};

static LaunchConfig g_launch_config{256, 64};
static TimingMode g_timing_mode = TimingMode::kMedian;
static int g_timing_repeats = kDefaultTimingRepeats;

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
  size_t size = 0;
  std::vector<int> input;
};

#define TASK_MIN 2
#define TASK_DEFAULT 8

static void merge_sorted(const std::vector<int> &input, size_t start, size_t end)
{
  size_t m = start + (end - start) / 2;

}

// Keep this reference body empty until sorting is requested.
/*
  N = 14
  10 5 9 2 13 8 4 7 1 12 11 3 14 6  

  i = 1, width = 2
  gx = 0, left = gx * 2, gx + 1 * 2 => (0, 2)
  left = 0, right = 2, mid = 0 + (2/2) = 1
  0,1,2
  num_slices = 14/2 = 7, need 7 threads/iterations

  merge_sorted(left, mid, right, input=input, output=output)
  5 10  9 2   8 13  4 7  1 12  3 11  6  14
  
  i= 2, width = 4
  gx * 4, gx + 1 * 4 = (0, 4)
  left = 0, right = 4, mid = left + (right - left) / 2 = 2
  0, 2, 4
  num_slices = 14 / 4 = 7/2 = 3.5, 3 full slices, 1 tail slice
  num_sliaces = (14 + 4 - 1)/2 = 17/2 = 4
  2 5 9 10  4 7 8 13  1 3 11 12  6  14

  i = 4, width = 8
  gx * 8, gx + 1 * 8 = (0, 8)
  left = 0, right = 8, mid = left + (right - left) / 2 = 4
  0, 4, 8
  2 4 5 7 8 9 10 13   1 3  6 11  12  14
  num_slices = 14/8 = 1.x slices i.e 1 full slice and .X slice
  num_slices = (14 + 8 - 1) / 2 = 21/8 = 2

  i = 8, width = 16
  left = 0, right = 16, m = l + (r - l)/2 = 16/2 = 8
  0, 8, max(N, right) +> 0, 8, 14
  num_slices = 14/16 => 0.x slices 0 full slice and .X slice
  num_slices = (14 + 16 -1)/16 = 29/16 = 1
  1  2  3 4 5 6 7 8 9 10 11 12 13 14

  i = 1 4 8 16  32  64
  merge = 2 8 16  32  64

*/

__host__ __device__ static void merge_sorted(
  int *input,
  int *output,
  size_t left,
  size_t mid,
  size_t right)
{
  int i = left;
  int j = mid;
  int k = left;
  while (i < mid && j < right)
  {
    const int i_val = input[i];
    const int j_val = input[j];

    if (i_val < j_val)
    {
      output[k] = i_val;
      i++;
    }
    else
    {
      output[k] = j_val;
      j++;
    }
    k++;
  }

  //merge left_part remainders
  while (i < mid)
  {
    output[k] = input[i];
    i++;
    k++;
  }

  //merge right part remainders
  while (j < right)
  {
    output[k] = input[j];
    j++;
    k++;
  }
}

static void cpu_array_sort_reference(std::vector<int>& input,
                                     std::vector<int>& output,
                                     size_t size) {
  std::vector<int> *inp = &input;
  std::vector<int> *oup = &output;
  
  for (size_t width = 1; width < size; width *= 2)
  {
    size_t span = width * 2;
    size_t num_slices = (size + span - 1) / span;

    for (size_t s = 0; s < num_slices; ++s)
    {
      size_t left  = s * span;
      size_t mid   = std::min(left + width, size);
      size_t right = std::min(left + span, size);
      merge_sorted((*inp).data(), (*oup).data(), left, mid, right);
    }

    std::vector<int> *temp = inp;
    inp = oup;
    oup = temp;
  }

  //at the end of each iteration, we swap input and output pointer
  //if the input pointer point to original ouput, then we know
  //in the last merge we merged original input into original output
  //so we are good, if not we have to memcpy back
  if (inp != &output && size != 0) {
    std::memcpy(output.data(), inp->data(), size * sizeof(int));
  }
}

// Keep this basic kernel body empty until sorting is requested.
__global__ void array_sort_basic_kernel(int* input, int* output,
                                        size_t size, size_t width) {
  size_t span  = width * 2;
  size_t gid   = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);
  size_t num_slices  = (size + span - 1) / span;

  for (size_t gx = gid; gx < num_slices; gx += grid_stride)
  {
    size_t  left  = gx * span;
    size_t  mid   = min(left + width, size);
    size_t  right = min(left + span, size);
    merge_sorted(input, output, left, mid, right);
  }
}

extern "C" void solution(int* input, int* output, size_t size) {
  const dim3 block_shape(g_launch_config.block_x, 1, 1);
  const dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  //kernel 1 basic
  //1(2), 2(4), 4(8), 8(16), 16(32)
  int *inp = input;
  int *oup = output;
  for (size_t width = 1; width < size; width *= 2)
  {
    array_sort_basic_kernel<<<grid_shape, block_shape>>>(inp, oup, size, width);
    int *temp = inp;
    inp       = oup;
    oup       = temp;
    //it think we might need to wait here before this kernel execution complete
    ///may be move the device syn here ?
  }
  CUDA_CHECK(cudaGetLastError());

  //at the end of each iteration, we swap input and output pointer
  //after swap, if input ptr points to orignal output, then we know
  //that the merge took place from input to output right before swap
  //if nowe have to copy data back.
  if (inp != output)
  {
    CUDA_CHECK(cudaMemcpy(output, inp, sizeof(int) * size,
                          cudaMemcpyDeviceToDevice));
  }
}

static std::vector<int> make_sort_input(size_t size) {
  std::vector<int> input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<int>((i * 37 + size * 11 + 19) % 4096) - 2048;
  }
  return input;
}

struct TestResult {
  std::string group;
  std::string name;
  size_t size = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

struct GpuJob {
  LaunchConfig launch_config;
  size_t result_index = 0;
  std::vector<float> samples;
};

static unsigned timing_seed(const TestCase& test, size_t job_count) {
  return kTimingShuffleSeed ^ static_cast<unsigned>(test.size) ^
         static_cast<unsigned>(job_count * 0x9E3779B9u);
}

static void launch_gpu_job(const GpuJob& job, int* device_input,
                           int* device_output, const TestCase& test) {
  g_launch_config = job.launch_config;
  solution(device_input, device_output, test.size);
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

static bool run_gpu_jobs(const TestCase& test,
                         const std::vector<int>& input,
                         std::vector<GpuJob>& jobs,
                         std::vector<TestResult>& results,
                         const std::vector<int>* reference) {
  int* device_input = nullptr;
  int* device_output = nullptr;
  const size_t bytes = input.size() * sizeof(int);
  std::vector<size_t> order;
  order.reserve(jobs.size());
  for (size_t i = 0; i < jobs.size(); ++i) {
    order.push_back(i);
  }

  CUDA_CHECK(cudaMalloc(&device_input, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes,
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
    std::vector<int> output(input.size(), 0);
    for (const GpuJob& job : jobs) {
      launch_gpu_job(job, device_input, device_output, test);
      CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes,
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
  std::cout << std::left << std::setw(10) << "group" << std::setw(24)
            << "name" << std::setw(12) << "size" << std::setw(8)
            << "block_x" << std::setw(8) << "grid_x" << std::setw(8)
            << "cpu" << std::setw(8) << "gpu" << std::setw(12)
            << "total_ms" << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(110, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);
  for (const auto& result : results) {
    std::cout << std::left << std::setw(10) << result.group
              << std::setw(24) << result.name << std::setw(12)
              << result.size << std::setw(8) << result.block_x
              << std::setw(8) << result.grid_x << std::setw(8)
              << result.cpu << std::setw(8) << result.gpu
              << std::setw(12) << result.total_ms << std::setw(12)
              << result.kernel_ms << '\n';
  }
}

static void print_scale_summary(const std::vector<TestResult>& results) {
  std::cout << "\nSweep summary (" << timing_mode_name()
            << " kernel_ms, lower is better)\n";
  for (const auto& result : results) {
    if (result.group == "sweep" && result.block_x == 256 &&
        result.grid_x == 64) {
      std::cout << "  " << result.name << ": " << result.kernel_ms
                << " ms at (" << result.block_x << ", " << result.grid_x
                << ")\n";
    }
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
      {"small_8", 8, {7, -2, 7, 1, 0, -9, 4, 1}},
      {"small_17", 17, make_sort_input(17)},
  };
  const std::vector<TestCase> published_tests = {
      {"tensara_1", 16384, {}},
      {"tensara_2", 32768, {}},
      {"tensara_3", 65536, {}},
      {"tensara_4", 131072, {}},
      {"tensara_5", 262144, {}},
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
                       std::vector<int> input,
                       const std::vector<LaunchConfig>& configs) {
    std::vector<int> reference;
    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      reference.assign(input.size(), 0);
      cpu_array_sort_reference(input, reference, test.size);
    }

    std::vector<GpuJob> jobs;
    for (const LaunchConfig& config : configs) {
      TestResult result;
      result.group = group;
      result.name = test.name;
      result.size = test.size;
      result.block_x = config.block_x;
      result.grid_x = config.grid_x;
      result.cpu = reference.empty() ? "SKIP" : "REF";
      result.gpu = "SKIP";
      results.push_back(result);
      jobs.push_back({config, results.size() - 1, {}});
    }

    const std::vector<int>* target = reference.empty() ? nullptr : &reference;
    if (kGpuKernelImplemented) {
      all_ok &= run_gpu_jobs(test, input, jobs, results, target);
    }
  };

  const std::vector<LaunchConfig> default_configs = {{256, 64}};
  if (!skip_cpu_verify) {
    for (const auto& test : exact_tests) {
      std::vector<int> input = test.input;
      run_group("exact", test, std::move(input), default_configs);
    }
  }

  if (skip_cpu_verify) {
    for (const auto& test : published_tests) {
      run_group("sweep", test, make_sort_input(test.size), sweep_configs);
    }
  }

  print_results(results);
  print_scale_summary(results);
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
