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
 * Problem 10: Sum Over Dimension
 * Source: https://tensara.org/problems/sum-dim
 *
 * Sum a row-major tensor over one dimension, keeping that dimension with
 * size one in the output. The input shape is arbitrary and dim is zero-based.
 *
 * Published shapes:
 * - (16, 128, 256), dim=1
 * - (32, 512, 512), dim=0
 * - (8, 1024, 1024), dim=2
 * - (64, 128, 128, 128), dim=2
 * - (4, 256, 256, 256), dim=1
 * - (128, 64, 64, 64), dim=3
 */

// The shape pointer is host metadata; input and output are device pointers.
// The output has the same rank as input, with shape[dim] replaced by one.
extern "C" void solution(const float* input, float* output, int dim,
                         const size_t* shape, size_t ndim);

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
  kBasicBlock,
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
    case KernelVariant::kBasicBlock:
      return "basic_block";
  }
  return "unknown";
}

static bool kernel_enabled(KernelVariant variant) {
  return !g_kernel_arg_set || g_kernel_variant == variant;
}

static const char* timing_mode_name() {
  return g_timing_mode == TimingMode::kMedian ? "median" : "best";
}

static float select_timing_sample(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  return g_timing_mode == TimingMode::kBest
             ? samples.front()
             : samples[samples.size() / 2];
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
  if (value == "basic_block") {
    g_kernel_variant = KernelVariant::kBasicBlock;
    g_kernel_arg_set = true;
    return true;
  }

  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic or --kernel=basic_block)\n";
  return false;
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
  const char* name;
  int dim;
  std::vector<size_t> shape;
};

struct ReductionParams {
  size_t outer_stride = 1;
  size_t reduce_dim = 1;
  size_t inner_stride = 1;
};

static ReductionParams make_reduction_params(int dim, const size_t* shape,
                                             size_t ndim) {
  ReductionParams params;
  for (int i = 0; i < dim; ++i) {
    params.outer_stride *= shape[i];
  }
  for (size_t i = static_cast<size_t>(dim + 1); i < ndim; ++i) {
    params.inner_stride *= shape[i];
  }
  params.reduce_dim = shape[dim];
  return params;
}

// CPU reference stub. Keep this body empty until the reference is requested.
static void cpu_sum_dim_reference(
    const std::vector<float>& input, std::vector<float>& output,
    size_t outer_stride, size_t reduce_dim, size_t inner_stride)
{
  size_t total_output_elems = outer_stride * inner_stride;

  for (size_t i = 0; i < total_output_elems; ++i)
  {
    size_t outer_pos = i / inner_stride;
    size_t inner_pos = i % inner_stride;

    float sum = 0.0f;
    for (size_t k = 0; k < reduce_dim; ++k)
    {
      size_t load_idx = (outer_pos * reduce_dim * inner_stride) + (k * inner_stride) + inner_pos;
      sum += input[load_idx];
    }
    output[i] = sum;
  }
}

// Basic GPU kernel stub. Keep this body empty until the kernel is requested.
__global__ void sum_dim_basic_kernel(const float* input, float* output,
                                     size_t outer_stride, size_t reduce_dim,
                                     size_t inner_stride)
{
  size_t total_output_elems = outer_stride * inner_stride;
  size_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gid; gx < total_output_elems; gx += grid_stride)
  {
    size_t outer_pos = gx / inner_stride;
    size_t inner_pos = gx % inner_stride;

    float sum = 0.0f;
    for (size_t k = 0; k < reduce_dim; ++k)
    {
      //jump over entire building i.e each building has rooms
      //each rooms have aisles, each aisles have books
      size_t l3_idx = (outer_pos * reduce_dim * inner_stride);
      //find the room/aisle i.e dimension trying to collapse
      size_t l2_idx = (k * inner_stride);
      //fine the specific book within that aisle.
      size_t l1_idx = inner_pos;
      sum += input[l3_idx + l2_idx + l1_idx];
    }
    output[gx] = sum;
  }
}

// Basic block GPU kernel stub. Keep this body empty until implemented.
__global__ void sum_dim_basic_block_kernel(
    const float* input, float* output, size_t outer_stride,
    size_t reduce_dim, size_t inner_stride) {

  size_t total_blocks = outer_stride;
  
  for (size_t bx = blockIdx.x; bx < total_blocks; bx += gridDim.x)
  {
    //block base
    //from input pespective , how many elemtns ot skip over ?
    size_t block_base = bx * reduce_dim * inner_stride;
    
    for (size_t lx = threadIdx.x; lx < inner_stride; lx += blockDim.x)
    {
      float sum = 0.0f; 
      for (int r = 0; r < reduce_dim; ++r)
      {
        size_t flat_idx = block_base + (r * inner_stride) + lx;
        sum += input[flat_idx];
      }

      //gid
      //from output prespective how many lements to skip over ?
      size_t gid = (bx * inner_stride) + lx;
      output[gid] = sum;
    }
  }
}

extern "C" void solution(const float* input, float* output, int dim,
                         const size_t* shape, size_t ndim) {
  const dim3 block_shape(g_launch_config.block_x, 1, 1);
  const dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  // Determine the number of elements in the dimensions before dim.
  size_t outer_stride = 1;
  for (int i = 0; i < dim; ++i)
    outer_stride *= shape[i];

  // Determine the number of elements in the dimensions after dim.
  size_t inner_stride = 1;
  for (int i = dim + 1; i < ndim; ++i)
    inner_stride *= shape[i];

  // The extent of the dimension being reduced.
  size_t reduce_dim = shape[dim];

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      sum_dim_basic_kernel<<<grid_shape, block_shape>>>(
          input, output, outer_stride, reduce_dim, inner_stride);
      break;
    case KernelVariant::kBasicBlock:
      sum_dim_basic_block_kernel<<<grid_shape, block_shape>>>(
          input, output, outer_stride, reduce_dim, inner_stride);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

static size_t element_count(const std::vector<size_t>& shape) {
  size_t count = 1;
  for (const size_t extent : shape) {
    count *= extent;
  }
  return count;
}

static std::vector<float> make_input(size_t count) {
  std::vector<float> input(count);
  for (size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
  }
  return input;
}

static size_t output_element_count(const TestCase& test) {
  return element_count(test.shape) / test.shape[test.dim];
}

static void run_cpu_reference(const TestCase& test,
                              const std::vector<float>& input,
                              std::vector<float>& output) {
  const ReductionParams params = make_reduction_params(
      test.dim, test.shape.data(), test.shape.size());
  cpu_sum_dim_reference(input, output, params.outer_stride,
                        params.reduce_dim, params.inner_stride);
}

static std::vector<float> run_gpu_case(const TestCase& test,
                                       const std::vector<float>& input) {
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t input_bytes = input.size() * sizeof(float);
  const size_t output_elements = output_element_count(test);
  const size_t output_bytes = output_elements * sizeof(float);
  std::vector<float> output(output_elements, 0.0f);

  CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&device_output, output_bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes,
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < kTimingWarmupIterations; ++i) {
    solution(device_input, device_output, test.dim, test.shape.data(),
             test.shape.size());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> total_samples;
  std::vector<float> kernel_samples;

  for (int repeat = 0; repeat < g_timing_repeats; ++repeat) {
    CUDA_CHECK(cudaEventRecord(start));
    solution(device_input, device_output, test.dim, test.shape.data(),
             test.shape.size());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_samples.push_back(elapsed_ms);
    kernel_samples.push_back(elapsed_ms);
  }

  g_last_timing.total_ms = select_timing_sample(total_samples);
  g_last_timing.kernel_ms = select_timing_sample(kernel_samples);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return output;
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
  size_t first_bad = 0;
  bool ok = true;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) {
      if (ok) {
        first_bad = i;
      }
      ok = false;
      continue;
    }

    const float diff = std::fabs(got[i] - expected[i]);
    if (diff > max_abs) {
      max_abs = diff;
      max_i = i;
    }
    const float tolerance = atol + rtol * std::fabs(expected[i]);
    if (diff > tolerance && ok) {
      first_bad = i;
      ok = false;
    }
  }

  if (!ok) {
    std::cerr << "verify(" << label << "): FAIL at i=" << first_bad
              << " got=" << got[first_bad]
              << " expected=" << expected[first_bad]
              << " max_abs=" << max_abs << " max_i=" << max_i << '\n';
  }
  return ok;
}

struct TestResult {
  const char* group = "";
  const char* name = "";
  const char* kernel = "";
  size_t ndim = 0;
  int dim = 0;
  size_t input_elements = 0;
  size_t output_elements = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(10) << "group" << std::setw(22)
            << "name" << std::setw(14) << "kernel" << std::setw(7)
            << "ndim" << std::setw(7) << "dim" << std::setw(14)
            << "input_elems" << std::setw(14)
            << "output_elems" << std::setw(8) << "block_x" << std::setw(8)
            << "grid_x" << std::setw(8) << "cpu" << std::setw(8) << "gpu"
            << std::setw(12) << "total_ms" << std::setw(12) << "kernel_ms"
            << '\n';
  std::cout << std::string(132, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& result : results) {
    std::cout << std::left << std::setw(10) << result.group
              << std::setw(22) << result.name << std::setw(7)
              << result.kernel << std::setw(7) << result.ndim << std::setw(7)
              << result.dim << std::setw(14)
              << result.input_elements << std::setw(14)
              << result.output_elements << std::setw(8) << result.block_x
              << std::setw(8) << result.grid_x << std::setw(8) << result.cpu
              << std::setw(8) << result.gpu << std::setw(12)
              << result.total_ms << std::setw(12) << result.kernel_ms << '\n';
  }
}

static void print_scale_summary(const std::vector<TestResult>& results) {
  std::vector<std::pair<std::string, std::string>> variants;
  for (const auto& result : results) {
    if (std::string(result.group) != "scale") {
      continue;
    }
    const std::pair<std::string, std::string> key(result.name, result.kernel);
    if (std::find(variants.begin(), variants.end(), key) == variants.end()) {
      variants.push_back(key);
    }
  }

  if (variants.empty()) {
    return;
  }

  std::cout << "\nScaling summary (" << timing_mode_name()
            << " kernel_ms, lower is better)\n";
  for (const auto& variant : variants) {
    const TestResult* best = nullptr;
    for (const auto& result : results) {
      if (std::string(result.group) == "scale" &&
          result.name == variant.first && result.kernel == variant.second &&
          (best == nullptr || result.kernel_ms < best->kernel_ms)) {
        best = &result;
      }
    }
    if (best != nullptr) {
      std::cout << "  " << variant.second << " / " << variant.first << ": "
                << best->kernel_ms << " ms at (" << best->block_x << ", "
                << best->grid_x << ")\n";
    }
  }
}

static int run_tests(bool skip_cpu_verify) {
  if (kGpuKernelImplemented && !cuda_runtime_ready()) {
    return 1;
  }

  std::cout << "Timing samples: mode=" << timing_mode_name()
            << " repeats=" << g_timing_repeats
            << " warmup=" << kTimingWarmupIterations
            << " metric=kernel_ms\n";
  std::cout << "CPU reference implemented: "
            << (kCpuReferenceImplemented ? "yes" : "no") << '\n';
  std::cout << "GPU kernel implemented: "
            << (kGpuKernelImplemented ? "yes" : "no") << "\n\n";

  const LaunchConfig default_launch = g_launch_config;
  const std::vector<TestCase> exact_tests = {
      {"small_axis_0", 0, {2, 3, 4}},
      {"small_axis_1", 1, {2, 3, 4}},
      {"small_axis_2", 2, {2, 3, 4}},
      {"small_4d_axis_2", 2, {2, 2, 2, 3}},
  };
  const std::vector<TestCase> medium_tests = {
      {"medium_axis_0", 0, {3, 17, 33}},
      {"medium_axis_1", 1, {8, 31, 65}},
      {"medium_axis_2", 2, {4, 32, 257}},
      {"medium_4d", 2, {3, 4, 8, 17}},
  };
  const std::vector<TestCase> large_tests = {
      {"large_axis_0", 0, {17, 64, 65}},
      {"large_axis_1", 1, {8, 257, 513}},
      {"large_4d", 1, {4, 32, 64, 33}},
  };
  const std::vector<TestCase> published_tests = {
      {"tensara_1", 1, {16, 128, 256}},
      {"tensara_2", 0, {32, 512, 512}},
      {"tensara_3", 2, {8, 1024, 1024}},
      {"tensara_4", 2, {64, 128, 128, 128}},
      {"tensara_5", 1, {4, 256, 256, 256}},
      {"tensara_6", 3, {128, 64, 64, 64}},
  };
  const std::vector<TestCase> shape_tests = {
      {"shape_3d_axis_0", 0, {5, 7, 9}},
      {"shape_3d_axis_2", 2, {5, 7, 9}},
      {"shape_4d_axis_1", 1, {3, 5, 7, 9}},
      {"shape_4d_axis_3", 3, {3, 5, 7, 9}},
  };
  const std::vector<TestCase> tail_tests = {
      {"tail_3d", 1, {7, 33, 65}},
      {"tail_4d", 2, {3, 17, 31, 65}},
      {"tail_axis_0", 0, {33, 5, 17}},
  };
  const std::vector<TestCase> cpu_scale_tests = {
      {"scale_tail_3d", 1, {5, 257, 513}},
      {"scale_tail_4d", 2, {3, 17, 65, 129}},
  };
  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};
  const KernelVariant kernel_variants[] = {KernelVariant::kBasic,
                                           KernelVariant::kBasicBlock};

  std::vector<TestResult> results;
  bool all_ok = true;

  auto run_case = [&](const char* group, const TestCase& test,
                      const std::vector<float>& input,
                      const std::vector<float>* expected) {
    const size_t input_elements = input.size();
    const size_t output_elements = output_element_count(test);
    std::vector<float> reference;
    std::string cpu_status = "SKIP";
    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      reference.assign(output_elements, 0.0f);
      run_cpu_reference(test, input, reference);
      cpu_status = "REF";
      if (expected != nullptr) {
        const bool cpu_ok = verify_close(reference, *expected, 1e-5f,
                                         1e-5f, test.name);
        cpu_status = cpu_ok ? "PASS" : "FAIL";
        all_ok &= cpu_ok;
      }
    }

    const std::vector<float> gpu = run_gpu_case(test, input);
    std::string gpu_status = "SKIP";
    if (expected != nullptr) {
      const bool gpu_ok = verify_close(gpu, *expected, 1e-5f, 1e-5f,
                                       test.name);
      gpu_status = gpu_ok ? "PASS" : "FAIL";
      all_ok &= gpu_ok;
    } else if (!skip_cpu_verify && kCpuReferenceImplemented) {
      const bool gpu_ok = verify_close(gpu, reference, 1e-5f, 1e-5f,
                                       test.name);
      gpu_status = gpu_ok ? "PASS" : "FAIL";
      all_ok &= gpu_ok;
    }

    results.push_back({group,
                       test.name,
                       current_kernel_name(),
                       test.shape.size(),
                       test.dim,
                       input_elements,
                       output_elements,
                       g_launch_config.block_x,
                       g_launch_config.grid_x,
                       cpu_status,
                       gpu_status,
                       g_last_timing.total_ms,
                       g_last_timing.kernel_ms});
  };

  auto run_sized = [&](const char* group, const TestCase& test) {
    g_launch_config = default_launch;
    const std::vector<float> input = make_input(element_count(test.shape));
    run_case(group, test, input, nullptr);
  };

  auto run_scale = [&](const TestCase& test) {
    const std::vector<float> input = make_input(element_count(test.shape));
    for (int block_x : scale_block_sizes) {
      for (int grid_x : scale_grid_sizes) {
        g_launch_config = {block_x, grid_x};
        run_case("scale", test, input, nullptr);
      }
    }
  };

  for (KernelVariant kernel_variant : kernel_variants) {
    if (!kernel_enabled(kernel_variant)) {
      continue;
    }
    g_kernel_variant = kernel_variant;

    for (const auto& test : exact_tests) {
      const size_t input_elements = element_count(test.shape);
      std::vector<float> input(input_elements);
      for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i);
      }

      std::vector<float> expected;
      if (test.name == std::string("small_axis_0")) {
        expected = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34};
      } else if (test.name == std::string("small_axis_1")) {
        expected = {12, 15, 18, 21, 48, 51, 54, 57};
      } else if (test.name == std::string("small_axis_2")) {
        expected = {6, 22, 38, 54, 70, 86};
      } else {
        expected = {3, 5, 7, 15, 17, 19, 27, 29, 31, 39, 41, 43};
      }
      g_launch_config = default_launch;
      run_case("small", test, input, &expected);
    }

    for (const auto& test : medium_tests) {
      run_sized("medium", test);
    }
    for (const auto& test : large_tests) {
      run_sized("large", test);
    }
    for (const auto& test : published_tests) {
      run_sized("tensara", test);
    }
    for (const auto& test : shape_tests) {
      run_sized("shape", test);
    }
    for (const auto& test : tail_tests) {
      run_sized("tail", test);
    }

    if (!skip_cpu_verify) {
      for (const auto& test : cpu_scale_tests) {
        run_scale(test);
      }
    } else {
      for (const auto& test : published_tests) {
        run_scale(test);
      }
    }
  }

  g_kernel_variant = KernelVariant::kBasic;
  g_launch_config = default_launch;
  print_results_table(results);
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
                << " [--skip-cpu] [--kernel=NAME]"
                << " [--timing=median|best]"
                << " [--timing-repeats=N]\n";
      return 0;
    } else if (arg == "--list-kernels") {
      std::cout << "basic\nbasic_block\n";
      return 0;
    } else if (arg.rfind("--kernel=", 0) == 0) {
      if (!parse_kernel_arg(arg)) {
        return 1;
      }
    } else if (arg.rfind("--timing=", 0) == 0) {
      const std::string value = arg.substr(9);
      if (value == "median") {
        g_timing_mode = TimingMode::kMedian;
      } else if (value == "best" || value == "min") {
        g_timing_mode = TimingMode::kBest;
      } else {
        std::cerr << "Unknown timing mode: " << value
                  << " (use --timing=median or --timing=best)\n";
        return 1;
      }
    } else if (arg.rfind("--timing-repeats=", 0) == 0) {
      try {
        size_t end = 0;
        const std::string value = arg.substr(17);
        const int repeats = std::stoi(value, &end);
        if (end != value.size() || repeats < 1) {
          throw std::invalid_argument("bad repeats");
        }
        g_timing_repeats = repeats;
      } catch (const std::exception&) {
        std::cerr << "Invalid timing repeat count\n";
        return 1;
      }
    } else {
      std::cerr << "Unknown argument: " << arg
                << " (supported: --skip-cpu, --kernel=..., "
                << "--timing=..., --timing-repeats=..., --list-kernels)\n";
      return 1;
    }
  }
  return run_tests(skip_cpu_verify);
}
