#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Tensara signature:
// - input, output are device pointers
// - input/output are row-major matrices with shape (m, n)
extern "C" void solution(const float* input, float* output, size_t n,
                         size_t m);

static constexpr bool kCpuReferenceImplemented = true;

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

static LaunchConfig g_launch_config{256, 64};
static KernelVariant g_kernel_variant = KernelVariant::kBasic;
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
  const char* name;
  size_t rows = 0;
  size_t cols = 0;
  std::vector<float> input;
  std::vector<float> expected;
};

static std::vector<float> cpu_relu(const std::vector<float>& input) {
  std::vector<float> output(input.size(), 0.0f);
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = input[i] < 0 ? 0.0f : input[i];
  }
  return output;
}

static std::vector<float> make_relu_input(size_t rows, size_t cols) {
  const size_t total = rows * cols;
  std::vector<float> input(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 37 + rows * 11 + cols * 7 + 17) % 257) - 128;
    float x = static_cast<float>(raw) / 19.0f;
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
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group" << std::setw(14) << "name"
            << std::setw(14) << "kernel" << std::setw(10) << "rows"
            << std::setw(10) << "cols" << std::setw(8) << "block_x"
            << std::setw(8) << "grid_x" << std::setw(6) << "cpu"
            << std::setw(6) << "gpu" << std::setw(12) << "total_ms"
            << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(108, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& r : results) {
    std::cout << std::left << std::setw(8) << r.group << std::setw(14)
              << r.name << std::setw(14) << r.kernel << std::setw(10)
              << r.rows << std::setw(10) << r.cols << std::setw(8)
              << r.block_x << std::setw(8) << r.grid_x << std::setw(6)
              << r.cpu << std::setw(6) << r.gpu << std::setw(12)
              << r.total_ms << std::setw(12) << r.kernel_ms << '\n';
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

static void run_solution_host(const std::vector<float>& input,
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

  CUDA_CHECK(cudaEventRecord(kernel_start));
  solution(d_input, d_output, cols, rows);
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  CUDA_CHECK(
      cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));

  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
  g_last_timing.total_ms = total_ms;
  g_last_timing.kernel_ms = kernel_ms;

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

static int run_tests(bool skip_cpu_verify) {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const LaunchConfig default_launch = g_launch_config;

  const std::vector<TestCase> tests = {
      {"small_1",
       2,
       4,
       {-1.0f, 0.0f, 1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f},
       {0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f}},
      {"small_2",
       3,
       3,
       {-7.5f, -0.5f, 0.0f, 0.5f, 2.5f, -3.25f, 9.0f, -9.0f, 1.25f},
       {0.0f, 0.0f, 0.0f, 0.5f, 2.5f, 0.0f, 9.0f, 0.0f, 1.25f}},
      {"small_3",
       1,
       8,
       {-8.0f, -4.0f, -0.0f, 0.0f, 0.25f, 1.0f, 8.0f, -2.0f},
       {0.0f, 0.0f, -0.0f, 0.0f, 0.25f, 1.0f, 8.0f, 0.0f}},
      {"small_tail2",
       2,
       3,
       {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -3.0f},
       {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f}},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } medium_tests[] = {
      {"medium_1", 64, 64},
      {"medium_2", 255, 257},
      {"medium_3", 513, 1025},
      {"medium_4", 1024, 1024},
      {"medium_tail2", 257, 258},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } large_verify_tests[] = {
      {"large_1", 1023, 2049},
      {"large_2", 1537, 2049},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } tensara_tests[] = {
      {"tensara_1", 4096, 4096},
      {"tensara_2", 6144, 4096},
      {"tensara_3", 4096, 7168},
      {"tensara_4", 4096, 8192},
      {"tensara_5", 8192, 8192},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } shape_tests[] = {
      {"shape_1", 6144, 4096},
      {"shape_2", 4096, 7168},
      {"shape_3", 4096, 8192},
      {"shape_4", 8192, 8192},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } tail_tests[] = {
      {"tail_1", 2049, 2049},
      {"tail_2", 3073, 4097},
      {"tail_3", 4097, 8193},
  };

  const struct {
    const char* name;
    size_t rows;
    size_t cols;
  } scale_tests[] = {
      {"scale_sq", 4096, 4096},
      {"scale_rect_1", 6144, 4096},
      {"scale_rect_2", 4096, 8192},
  };

  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};
  const KernelVariant kernel_variants[] = {KernelVariant::kBasic,
                                           KernelVariant::kFloat4};

  bool all_ok = true;
  std::vector<TestResult> results;

  auto run_sized = [&](const char* group, const char* name, size_t rows,
                       size_t cols) {
    g_launch_config = default_launch;

    const auto input = make_relu_input(rows, cols);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref = cpu_relu(input);
      cpu_status = "REF";
    }

    std::vector<float> gpu_out(input.size(), 0.0f);
    run_solution_host(input, gpu_out, cols, rows);

    TestResult res;
    res.group = group;
    res.name = name;
    res.kernel = current_kernel_name();
    res.rows = rows;
    res.cols = cols;
    res.block_x = g_launch_config.block_x;
    res.grid_x = g_launch_config.grid_x;
    res.cpu = cpu_status;

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      const bool gpu_ok = verify_close(gpu_out, ref, 1e-6f, 1e-6f, name, false);
      all_ok &= gpu_ok;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
    } else {
      res.gpu = "SKIP";
    }

    res.total_ms = g_last_timing.total_ms;
    res.kernel_ms = g_last_timing.kernel_ms;
    results.push_back(res);
  };

  auto run_scaling = [&](const char* name, size_t rows, size_t cols) {
    const auto input = make_relu_input(rows, cols);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref = cpu_relu(input);
      cpu_status = "REF";
    }

    for (int block_x : scale_block_sizes) {
      for (int grid_x : scale_grid_sizes) {
        g_launch_config = {block_x, grid_x};

        std::vector<float> gpu_out(input.size(), 0.0f);
        run_solution_host(input, gpu_out, cols, rows);

        TestResult res;
        res.group = "scale";
        res.name = name;
        res.kernel = current_kernel_name();
        res.rows = rows;
        res.cols = cols;
        res.block_x = g_launch_config.block_x;
        res.grid_x = g_launch_config.grid_x;
        res.cpu = cpu_status;

        if (!skip_cpu_verify && kCpuReferenceImplemented) {
          const bool gpu_ok =
              verify_close(gpu_out, ref, 1e-6f, 1e-6f, name, false);
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
    g_kernel_variant = kernel_variant;

    for (const auto& tc : tests) {
      g_launch_config = default_launch;

      std::string cpu_status = "SKIP";
      if (!skip_cpu_verify && kCpuReferenceImplemented) {
        const auto ref = cpu_relu(tc.input);
        const bool cpu_ok =
            verify_close(ref, tc.expected, 1e-6f, 1e-6f, tc.name, false);
        cpu_status = cpu_ok ? "PASS" : "FAIL";
        all_ok &= cpu_ok;
      }

      std::vector<float> gpu_out(tc.input.size(), 0.0f);
      run_solution_host(tc.input, gpu_out, tc.cols, tc.rows);

      const bool gpu_ok =
          verify_close(gpu_out, tc.expected, 1e-6f, 1e-6f, tc.name, false);
      all_ok &= gpu_ok;

      TestResult res;
      res.group = "small";
      res.name = tc.name;
      res.kernel = current_kernel_name();
      res.rows = tc.rows;
      res.cols = tc.cols;
      res.block_x = g_launch_config.block_x;
      res.grid_x = g_launch_config.grid_x;
      res.cpu = cpu_status;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
      res.total_ms = g_last_timing.total_ms;
      res.kernel_ms = g_last_timing.kernel_ms;
      results.push_back(res);
    }

    for (const auto& mt : medium_tests) {
      run_sized("medium", mt.name, mt.rows, mt.cols);
    }

    for (const auto& lt : large_verify_tests) {
      run_sized("large", lt.name, lt.rows, lt.cols);
    }

    if (!skip_cpu_verify) {
      run_scaling("scale_tail2", 257, 258);
      run_scaling("scale_rect", 513, 1025);
    }

    if (skip_cpu_verify) {
      for (const auto& tt : tensara_tests) {
        run_sized("tensara", tt.name, tt.rows, tt.cols);
      }
      for (const auto& st : shape_tests) {
        run_sized("shape", st.name, st.rows, st.cols);
      }
      for (const auto& tt : tail_tests) {
        run_sized("tail", tt.name, tt.rows, tt.cols);
      }
      for (const auto& sc : scale_tests) {
        run_scaling(sc.name, sc.rows, sc.cols);
      }
    }
  }

  g_kernel_variant = KernelVariant::kBasic;
  g_launch_config = default_launch;
  print_results_table(results);
  print_scale_heatmaps(results);
  return all_ok ? 0 : 1;
}

__global__ void device_relu_basic(const float* input, float* output,
                                  size_t total) {
  const size_t gix = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t gx = gix; gx < total; gx += grid_stride) {
    float val = input[gx];
    output[gx] = val < 0 ? 0.0f : val;
  }
}

__global__ void device_relu_float4(const float* input, float* output,
                                   size_t total) {

  //converts arrays to float4 array so each[i] represents block of 4
  //but can be treated like single unit
  size_t total_vec = total / 4;
  const float4 *input_vec = reinterpret_cast<const float4 *>(input);
  float4 *output_vec = reinterpret_cast<float4 *>(output);

  //each thread processes 4 elements
  const size_t gix = ((blockDim.x * blockIdx.x) + threadIdx.x);  

  //each grid processes 4 elements each
  const size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gix; gx < total_vec; gx += grid_stride)
  {
    float4 ivec = input_vec[gx];
    ivec.w = ivec.w < 0 ? 0.0f : ivec.w;
    ivec.x = ivec.x < 0 ? 0.0f : ivec.x;
    ivec.y = ivec.y < 0 ? 0.0f : ivec.y;
    ivec.z = ivec.z < 0 ? 0.0f : ivec.z;
    output_vec[gx] = ivec;
  }

  //tail part
  size_t tail_start = (total_vec * 4) + gix;
  for (size_t gx = tail_start; gx < total; gx += grid_stride)
  {
    float val = input[gx];
    output[gx] = val < 0 ? 0.0f : val;
  }
}

extern "C" void solution(const float* input, float* output, size_t n,
                         size_t m) {
  const size_t total = n * m;
  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      device_relu_basic<<<grid_shape, block_shape>>>(input, output, total);
      break;
    case KernelVariant::kFloat4:
      device_relu_float4<<<grid_shape, block_shape>>>(input, output, total);
      break;
  }

  CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  bool skip_cpu_verify = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--skip-cpu") {
      skip_cpu_verify = true;
    } else {
      std::cerr << "Unknown argument: " << argv[i]
                << " (supported: --skip-cpu)\n";
      return 1;
    }
  }

  return run_tests(skip_cpu_verify);
}
