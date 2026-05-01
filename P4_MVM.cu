#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Tensara-style signature:
// - input_a, input_b, output_c are device pointers
// - input_a is a row-major matrix with shape (m, k)
// - input_b is a vector with shape (k)
// - output_c is a vector with shape (m)
extern "C" void solution(const float* input_a, const float* input_b,
                         float* output_c, size_t m, size_t k);

// Flip this to true after implementing cpu_matrix_vector().
static constexpr bool kCpuReferenceImplemented = false;

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
};

static LaunchConfig g_launch_config{256, 64};
static KernelVariant g_kernel_variant = KernelVariant::kBasic;
static Timing g_last_timing;

static const char* current_kernel_name() {
  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      return "basic";
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
  const char* name = "";
  size_t m = 0;
  size_t k = 0;
  std::vector<float> input_a;
  std::vector<float> input_b;
  std::vector<float> expected;
};

// CPU reference stub.
//
// input_a: row-major matrix with shape (m, k), stored as input_a[row * k + col]
// input_b: vector with shape (k)
// output_c: vector with shape (m), where output_c[row] = input_a[row, :] dot input_b
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
static void cpu_matrix_vector(
    const std::vector<float>& input_a,
    const std::vector<float>& input_b,
    std::vector<float>& output_c,
    size_t m,
    size_t k)
{

}

static std::vector<float> make_matrix_input(size_t m, size_t k) {
  const size_t total = m * k;
  std::vector<float> input_a(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 37 + m * 11 + k * 7 + 13) % 251) - 125;
    float x = static_cast<float>(raw) / 31.0f;
    if (i % 29 == 0) {
      x = 0.0f;
    } else if (i % 17 == 0) {
      x = -std::fabs(x);
    }
    input_a[i] = x;
  }
  return input_a;
}

static std::vector<float> make_vector_input(size_t k) {
  std::vector<float> input_b(k, 0.0f);
  for (size_t i = 0; i < k; ++i) {
    const int raw = static_cast<int>((i * 19 + k * 5 + 23) % 127) - 63;
    float x = static_cast<float>(raw) / 29.0f;
    if (i % 31 == 0) {
      x = 0.0f;
    } else if (i % 11 == 0) {
      x = -std::fabs(x);
    }
    input_b[i] = x;
  }
  return input_b;
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
  size_t m = 0;
  size_t k = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group" << std::setw(16) << "name"
            << std::setw(12) << "kernel" << std::setw(10) << "m"
            << std::setw(10) << "k" << std::setw(8) << "block_x"
            << std::setw(8) << "grid_x" << std::setw(6) << "cpu"
            << std::setw(6) << "gpu" << std::setw(12) << "total_ms"
            << std::setw(12) << "kernel_ms" << '\n';
  std::cout << std::string(108, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& r : results) {
    std::cout << std::left << std::setw(8) << r.group << std::setw(16)
              << r.name << std::setw(12) << r.kernel << std::setw(10) << r.m
              << std::setw(10) << r.k << std::setw(8) << r.block_x
              << std::setw(8) << r.grid_x << std::setw(6) << r.cpu
              << std::setw(6) << r.gpu << std::setw(12) << r.total_ms
              << std::setw(12) << r.kernel_ms << '\n';
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

static void run_solution_host(const std::vector<float>& input_a,
                              const std::vector<float>& input_b,
                              std::vector<float>& output_c, size_t m,
                              size_t k) {
  cudaEvent_t total_start = nullptr;
  cudaEvent_t total_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  const size_t bytes_a = m * k * sizeof(float);
  const size_t bytes_b = k * sizeof(float);
  const size_t bytes_c = m * sizeof(float);

  float* d_input_a = nullptr;
  float* d_input_b = nullptr;
  float* d_output_c = nullptr;

  CUDA_CHECK(cudaEventRecord(total_start));
  CUDA_CHECK(cudaMalloc(&d_input_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_input_b, bytes_b));
  CUDA_CHECK(cudaMalloc(&d_output_c, bytes_c));
  CUDA_CHECK(cudaMemcpy(d_input_a, input_a.data(), bytes_a,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input_b, input_b.data(), bytes_b,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output_c, 0, bytes_c));

  CUDA_CHECK(cudaEventRecord(kernel_start));
  solution(d_input_a, d_input_b, d_output_c, m, k);
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  CUDA_CHECK(cudaMemcpy(output_c.data(), d_output_c, bytes_c,
                        cudaMemcpyDeviceToHost));
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
  CUDA_CHECK(cudaFree(d_input_a));
  CUDA_CHECK(cudaFree(d_input_b));
  CUDA_CHECK(cudaFree(d_output_c));
}

static int run_tests(bool skip_cpu_verify) {
  if (!cuda_runtime_ready()) {
    return 1;
  }

  const LaunchConfig default_launch = g_launch_config;
  const KernelVariant kernel_variants[] = {KernelVariant::kBasic};

  const std::vector<TestCase> tests = {
      {"small_1",
       2,
       3,
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
       {1.0f, 0.0f, -1.0f},
       {-2.0f, -2.0f}},
      {"small_2",
       3,
       1,
       {-2.0f, 0.0f, 3.5f},
       {2.0f},
       {-4.0f, 0.0f, 7.0f}},
      {"small_3",
       1,
       4,
       {0.5f, -1.0f, 2.0f, -4.0f},
       {2.0f, -3.0f, 0.25f, -0.5f},
       {6.5f}},
      {"small_4",
       3,
       4,
       {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 1.5f, 0.5f, 0.0f, 1.0f,
        0.0f, 2.0f},
       {0.25f, -1.0f, 2.0f, 0.5f},
       {6.25f, 4.0f, 0.0f}},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } medium_tests[] = {
      {"medium_1", 64, 64},
      {"medium_2", 255, 257},
      {"medium_3", 513, 1025},
      {"medium_4", 1024, 1024},
      {"medium_tail", 257, 258},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } large_verify_tests[] = {
      {"large_1", 1023, 2049},
      {"large_2", 1537, 2049},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } tensara_tests[] = {
      {"tensara_1", 4096, 4096},
      {"tensara_2", 6144, 4096},
      {"tensara_3", 7168, 4096},
      {"tensara_4", 8192, 4096},
      {"tensara_5", 9216, 4096},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } shape_tests[] = {
      {"shape_tall", 8192, 1024},
      {"shape_wide", 1024, 8192},
      {"shape_odd", 4097, 2049},
      {"shape_rect", 2049, 4097},
  };

  const struct {
    const char* name;
    size_t m;
    size_t k;
  } scale_tests[] = {
      {"scale_sq", 4096, 4096},
      {"scale_tall", 8192, 2048},
      {"scale_wide", 2048, 8192},
  };

  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};

  bool all_ok = true;
  std::vector<TestResult> results;

  auto run_sized = [&](const char* group, const char* name, size_t m,
                       size_t k) {
    g_launch_config = default_launch;

    const auto input_a = make_matrix_input(m, k);
    const auto input_b = make_vector_input(k);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref.assign(m, 0.0f);
      cpu_matrix_vector(input_a, input_b, ref, m, k);
      cpu_status = "REF";
    }

    std::vector<float> gpu_out(m, 0.0f);
    run_solution_host(input_a, input_b, gpu_out, m, k);

    TestResult res;
    res.group = group;
    res.name = name;
    res.kernel = current_kernel_name();
    res.m = m;
    res.k = k;
    res.block_x = g_launch_config.block_x;
    res.grid_x = g_launch_config.grid_x;
    res.cpu = cpu_status;

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      const bool gpu_ok =
          verify_close(gpu_out, ref, 1e-3f, 1e-3f, name, false);
      all_ok &= gpu_ok;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
    } else {
      res.gpu = "SKIP";
    }

    res.total_ms = g_last_timing.total_ms;
    res.kernel_ms = g_last_timing.kernel_ms;
    results.push_back(res);
  };

  auto run_scaling = [&](const char* name, size_t m, size_t k) {
    const auto input_a = make_matrix_input(m, k);
    const auto input_b = make_vector_input(k);
    std::vector<float> ref;
    std::string cpu_status = "SKIP";

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      ref.assign(m, 0.0f);
      cpu_matrix_vector(input_a, input_b, ref, m, k);
      cpu_status = "REF";
    }

    for (int block_x : scale_block_sizes) {
      for (int grid_x : scale_grid_sizes) {
        g_launch_config = {block_x, grid_x};

        std::vector<float> gpu_out(m, 0.0f);
        run_solution_host(input_a, input_b, gpu_out, m, k);

        TestResult res;
        res.group = "scale";
        res.name = name;
        res.kernel = current_kernel_name();
        res.m = m;
        res.k = k;
        res.block_x = g_launch_config.block_x;
        res.grid_x = g_launch_config.grid_x;
        res.cpu = cpu_status;

        if (!skip_cpu_verify && kCpuReferenceImplemented) {
          const bool gpu_ok =
              verify_close(gpu_out, ref, 1e-3f, 1e-3f, name, false);
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
        std::vector<float> ref(tc.m, 0.0f);
        cpu_matrix_vector(tc.input_a, tc.input_b, ref, tc.m, tc.k);
        const bool cpu_ok =
            verify_close(ref, tc.expected, 1e-5f, 1e-5f, tc.name, false);
        cpu_status = cpu_ok ? "PASS" : "FAIL";
        all_ok &= cpu_ok;
      }

      std::vector<float> gpu_out(tc.m, 0.0f);
      run_solution_host(tc.input_a, tc.input_b, gpu_out, tc.m, tc.k);

      const bool gpu_ok =
          verify_close(gpu_out, tc.expected, 1e-4f, 1e-4f, tc.name, false);
      all_ok &= gpu_ok;

      TestResult res;
      res.group = "small";
      res.name = tc.name;
      res.kernel = current_kernel_name();
      res.m = tc.m;
      res.k = tc.k;
      res.block_x = g_launch_config.block_x;
      res.grid_x = g_launch_config.grid_x;
      res.cpu = cpu_status;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
      res.total_ms = g_last_timing.total_ms;
      res.kernel_ms = g_last_timing.kernel_ms;
      results.push_back(res);
    }

    for (const auto& mt : medium_tests) {
      run_sized("medium", mt.name, mt.m, mt.k);
    }

    for (const auto& lt : large_verify_tests) {
      run_sized("large", lt.name, lt.m, lt.k);
    }

    if (!skip_cpu_verify) {
      run_scaling("scale_verify", 257, 383);
    }

    if (skip_cpu_verify) {
      for (const auto& tt : tensara_tests) {
        run_sized("tensara", tt.name, tt.m, tt.k);
      }
      for (const auto& st : shape_tests) {
        run_sized("shape", st.name, st.m, st.k);
      }
      for (const auto& sc : scale_tests) {
        run_scaling(sc.name, sc.m, sc.k);
      }
    }
  }

  g_kernel_variant = KernelVariant::kBasic;
  g_launch_config = default_launch;
  print_results_table(results);
  print_scale_heatmaps(results);
  return all_ok ? 0 : 1;
}

// Basic GPU kernel stub.
//
// input_a: device pointer to a row-major matrix with shape (m, k),
//          stored as input_a[row * k + col]
// input_b: device pointer to a vector with shape (k)
// output_c: device pointer to a vector with shape (m)
// m: number of matrix rows and output elements
// k: number of matrix columns and vector elements
__global__ void device_mvm_basic(const float* input_a, const float* input_b,
                                 float* output_c, size_t m, size_t k) {


}

extern "C" void solution(const float* input_a, const float* input_b,
                         float* output_c, size_t m, size_t k) {
  dim3 block_shape(g_launch_config.block_x, 1, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      device_mvm_basic<<<grid_shape, block_shape>>>(input_a, input_b, output_c,
                                                    m, k);
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
