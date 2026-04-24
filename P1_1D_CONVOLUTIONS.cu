#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

void solution(const float* A, const float* B, float* C, size_t N, size_t K);

enum class KernelKind {
  Basic,
  BasicConst,
  Tiled,
  TiledConst,
  BlockStrideStub,
  BlockStrideConst,
};

static KernelKind g_kernel_kind = KernelKind::Basic;

struct Timing {
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static Timing g_last_timing;
static bool g_print_timing = true;

struct LaunchConfig {
  int block_x = 256;
  int grid_x = 32;
};

static LaunchConfig g_launch_config{256, 32};

constexpr size_t kMaxConstantFilterSize = 8192;
__constant__ float g_const_b[kMaxConstantFilterSize];

static bool kernel_uses_constant_filter(KernelKind kind) {
  switch (kind) {
    case KernelKind::BasicConst:
    case KernelKind::TiledConst:
    case KernelKind::BlockStrideConst:
      return true;
    default:
      return false;
  }
}

static bool kernel_supports_filter(KernelKind kind, size_t K) {
  if (!kernel_uses_constant_filter(kind)) {
    return true;
  }
  return K <= kMaxConstantFilterSize;
}

static bool parse_kernel_arg(const std::string& arg) {
  const std::string prefix = "--kernel=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  const std::string value = arg.substr(prefix.size());
  if (value == "basic") {
    g_kernel_kind = KernelKind::Basic;
    return true;
  }
  if (value == "basic-const") {
    g_kernel_kind = KernelKind::BasicConst;
    return true;
  }
  if (value == "tiled") {
    g_kernel_kind = KernelKind::Tiled;
    return true;
  }
  if (value == "tiled-const") {
    g_kernel_kind = KernelKind::TiledConst;
    return true;
  }
  if (value == "block-stride-stub") {
    g_kernel_kind = KernelKind::BlockStrideStub;
    return true;
  }
  if (value == "block-stride-const") {
    g_kernel_kind = KernelKind::BlockStrideConst;
    return true;
  }
  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic, --kernel=basic-const,"
            << " --kernel=tiled, --kernel=tiled-const,"
            << " --kernel=block-stride-stub,"
            << " --kernel=block-stride-const)\n";
  return false;
}

inline void cuda_check(cudaError_t err, const char* file, int line,
                       const char* expr) {
  if (err == cudaSuccess) {
    return;
  }
  std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << expr
            << ") at " << file << ":" << line << '\n';
}

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__, #expr)

struct TestCase {
  const char* name;
  std::vector<float> A;
  std::vector<float> B;
  std::vector<float> expected;
};

static std::vector<float> cpu_conv_same(const std::vector<float>& A,
                                        const std::vector<float>& B) {
  const size_t N = A.size();
  const size_t K = B.size();
  const size_t r = (K > 0) ? (K - 1) / 2 : 0;
  std::vector<float> C(N, 0.0f);
  for (size_t i = 0; i < N; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < K; ++j) {
      const long idx =
          static_cast<long>(i) + static_cast<long>(j) - static_cast<long>(r);
      if (idx >= 0 && static_cast<size_t>(idx) < N) {
        sum += A[static_cast<size_t>(idx)] * B[j];
      }
    }
    C[i] = sum;
  }
  return C;
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
  size_t N = 0;
  size_t K = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(8) << "group" << std::setw(12) << "name"
            << std::setw(10) << "kernel" << std::setw(10) << "N"
            << std::setw(8) << "K" << std::setw(8) << "block_x"
            << std::setw(8) << "grid_x" << std::setw(6) << "cpu"
            << std::setw(6) << "gpu" << std::setw(12) << "total_ms"
            << std::setw(12) << "kernel_ms"
            << '\n';
  std::cout << std::string(100, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);
  for (const auto& r : results) {
    std::cout << std::left << std::setw(8) << r.group << std::setw(12)
              << r.name << std::setw(10) << r.kernel << std::setw(10) << r.N
              << std::setw(8) << r.K << std::setw(8) << r.block_x
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
    if (r.group != std::string("scale")) {
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
            r.kernel == kernel) {
          if (best_ms < 0.0f || r.kernel_ms < best_ms) {
            best_ms = r.kernel_ms;
            best_block = r.block_x;
            best_grid = r.grid_x;
          }
        }
      }
      if (best_ms < 0.0f) {
        continue;
      }

      std::cout << '\n' << name << " / " << kernel << "  best=(" << best_block
                << "," << best_grid << ") " << best_ms << " ms\n";
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
                r.grid_x == grid_x) {
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

static int run_tests(bool skip_cpu_verify) {
  g_print_timing = false;
  const LaunchConfig default_launch = g_launch_config;
  const struct {
    KernelKind kind;
    const char* name;
  } kernels[] = {
      {KernelKind::Basic, "basic"},
      {KernelKind::BasicConst, "basic_c"},
      {KernelKind::Tiled, "tiled"},
      {KernelKind::TiledConst, "tiled_c"},
      {KernelKind::BlockStrideStub, "bstride"},
      {KernelKind::BlockStrideConst, "bstride_c"},
  };
  const std::vector<TestCase> tests = {
      {"small_1", {1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 1.0f},
       {4.0f, 8.0f, 12.0f, 11.0f}},
      {"small_2", {1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f},
       {3.0f, 6.0f, 5.0f}},
      {"small_3", {10.0f, 20.0f, 30.0f}, {1.0f, 2.0f, 1.0f},
       {40.0f, 80.0f, 80.0f}},
  };
  bool all_ok = true;
  std::vector<TestResult> results;

  for (const auto& tc : tests) {
    std::string cpu_status = "SKIP";
    if (!skip_cpu_verify) {
      const auto ref = cpu_conv_same(tc.A, tc.B);
      const bool cpu_ok =
          verify_close(ref, tc.expected, 1e-5f, 1e-5f, tc.name, false);
      cpu_status = cpu_ok ? "PASS" : "FAIL";
      all_ok &= cpu_ok;
    }
    for (const auto& k : kernels) {
      g_kernel_kind = k.kind;
      std::vector<float> gpu_out(tc.A.size(), 0.0f);
      solution(tc.A.data(), tc.B.data(), gpu_out.data(), tc.A.size(),
               tc.B.size());
      const bool gpu_ok =
          verify_close(gpu_out, tc.expected, 1e-4f, 1e-4f, tc.name, false);
      all_ok &= gpu_ok;
      TestResult res;
      res.group = "small";
      res.name = tc.name;
      res.kernel = k.name;
      res.N = tc.A.size();
      res.K = tc.B.size();
      res.block_x = g_launch_config.block_x;
      res.grid_x = g_launch_config.grid_x;
      res.cpu = cpu_status;
      res.gpu = gpu_ok ? "PASS" : "FAIL";
      res.total_ms = g_last_timing.total_ms;
      res.kernel_ms = g_last_timing.kernel_ms;
      results.push_back(res);
    }
  }
  const struct {
    const char* name;
    size_t N;
    size_t K;
  } large_tests[] = {
      {"large_1", 1u << 15, 31},
      {"large_2", 1u << 16, 63},
      {"large_3", 1u << 17, 95},
      {"large_4", 1u << 18, 127},
      {"large_5", 1u << 19, 191},
  };
  const struct {
    const char* name;
    size_t N;
    size_t K;
  } tile_tests[] = {
      {"tile_1", 1u << 18, 127},
      {"tile_2", 1u << 19, 191},
      {"tile_3", 1u << 20, 255},
      {"tile_4", 1u << 20, 383},
      {"tile_5", 1u << 21, 511},
  };
  const struct {
    const char* name;
    size_t N;
    size_t K;
  } web_tests[] = {
      {"web_1", 1u << 15, 8191},
      {"web_2", 1u << 16, 8191},
  };
  const int scale_block_sizes[] = {32, 64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64};
  auto run_sized = [&](const char* group, const char* name, size_t N,
                       size_t K) {
    g_launch_config = default_launch;
    std::vector<float> A(N);
    std::vector<float> B(K);
    for (size_t i = 0; i < N; ++i) {
      A[i] = static_cast<float>((i % 97) - 48) / 50.0f;
    }
    for (size_t j = 0; j < K; ++j) {
      B[j] = static_cast<float>((j % 11) - 5) / 7.0f;
    }
    std::string cpu_status = "SKIP";
    if (!skip_cpu_verify) {
      const auto ref = cpu_conv_same(A, B);
      cpu_status = "REF";
      for (const auto& k : kernels) {
        if (!kernel_supports_filter(k.kind, K)) {
          TestResult res;
          res.group = group;
          res.name = name;
          res.kernel = k.name;
          res.N = N;
          res.K = K;
          res.block_x = g_launch_config.block_x;
          res.grid_x = g_launch_config.grid_x;
          res.cpu = cpu_status;
          res.gpu = "SKIP";
          results.push_back(res);
          continue;
        }
        g_kernel_kind = k.kind;
        std::vector<float> gpu_out(N, 0.0f);
        solution(A.data(), B.data(), gpu_out.data(), N, K);
        const bool gpu_ok =
            verify_close(gpu_out, ref, 1e-3f, 1e-3f, name, false);
        all_ok &= gpu_ok;
        TestResult res;
        res.group = group;
        res.name = name;
        res.kernel = k.name;
        res.N = N;
        res.K = K;
        res.block_x = g_launch_config.block_x;
        res.grid_x = g_launch_config.grid_x;
        res.cpu = cpu_status;
        res.gpu = gpu_ok ? "PASS" : "FAIL";
        res.total_ms = g_last_timing.total_ms;
        res.kernel_ms = g_last_timing.kernel_ms;
        results.push_back(res);
      }
      return;
    }
    for (const auto& k : kernels) {
      if (!kernel_supports_filter(k.kind, K)) {
        TestResult res;
        res.group = group;
        res.name = name;
        res.kernel = k.name;
        res.N = N;
        res.K = K;
        res.block_x = g_launch_config.block_x;
        res.grid_x = g_launch_config.grid_x;
        res.cpu = cpu_status;
        res.gpu = "SKIP";
        results.push_back(res);
        continue;
      }
      g_kernel_kind = k.kind;
      std::vector<float> gpu_out(N, 0.0f);
      solution(A.data(), B.data(), gpu_out.data(), N, K);
      TestResult res;
      res.group = group;
      res.name = name;
      res.kernel = k.name;
      res.N = N;
      res.K = K;
      res.block_x = g_launch_config.block_x;
      res.grid_x = g_launch_config.grid_x;
      res.cpu = cpu_status;
      res.gpu = "SKIP";
      res.total_ms = g_last_timing.total_ms;
      res.kernel_ms = g_last_timing.kernel_ms;
      results.push_back(res);
    }
  };
  auto run_scaling = [&](const char* name, size_t N, size_t K) {
    std::vector<float> A(N);
    std::vector<float> B(K);
    for (size_t i = 0; i < N; ++i) {
      A[i] = static_cast<float>((i % 97) - 48) / 50.0f;
    }
    for (size_t j = 0; j < K; ++j) {
      B[j] = static_cast<float>((j % 11) - 5) / 7.0f;
    }

    std::vector<float> ref;
    std::string cpu_status = "SKIP";
    if (!skip_cpu_verify) {
      ref = cpu_conv_same(A, B);
      cpu_status = "REF";
    }

    for (int block_x : scale_block_sizes) {
      for (int grid_x : scale_grid_sizes) {
        g_launch_config = {block_x, grid_x};
        for (const auto& k : kernels) {
          if (!kernel_supports_filter(k.kind, K)) {
            TestResult res;
            res.group = "scale";
            res.name = name;
            res.kernel = k.name;
            res.N = N;
            res.K = K;
            res.block_x = g_launch_config.block_x;
            res.grid_x = g_launch_config.grid_x;
            res.cpu = cpu_status;
            res.gpu = "SKIP";
            results.push_back(res);
            continue;
          }
          g_kernel_kind = k.kind;
          std::vector<float> gpu_out(N, 0.0f);
          solution(A.data(), B.data(), gpu_out.data(), N, K);

          TestResult res;
          res.group = "scale";
          res.name = name;
          res.kernel = k.name;
          res.N = N;
          res.K = K;
          res.block_x = g_launch_config.block_x;
          res.grid_x = g_launch_config.grid_x;
          res.cpu = cpu_status;

          if (!skip_cpu_verify) {
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
    }
  };
  for (const auto& lt : large_tests) {
    run_sized("large", lt.name, lt.N, lt.K);
  }
  for (const auto& lt : tile_tests) {
    run_sized("tile", lt.name, lt.N, lt.K);
  }
  for (const auto& wt : web_tests) {
    run_sized("web", wt.name, wt.N, wt.K);
  }
  for (const auto& wt : web_tests) {
    run_scaling(wt.name, wt.N, wt.K);
  }
  g_launch_config = default_launch;
  print_results_table(results);
  print_scale_heatmaps(results);
  return all_ok ? 0 : 1;
}

__global__ void device_1d_conv_basic(
  const float *gm_a, 
  const float *gm_b, 
  float *gm_c, 
  size_t K, 
  size_t N)
{
  int glx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int grid_stride_x = (gridDim.x * blockDim.x);
  int r = (K - 1) / 2; 

  //grid stride loop
  for (int gx = glx; gx < N; gx += grid_stride_x)
  {
    //compute output
    float sum = 0.0f;
    for (int h = 0; h < K; ++h)
    {
      int a_idx = gx + h - r;
      if (0 <= a_idx && a_idx < N)
        sum += gm_a[a_idx] * gm_b[h];
    }
    gm_c[gx] = sum;
  }
}

__global__ void device_1d_conv_basic_const(
  const float *gm_a,
  float *gm_c,
  size_t K,
  size_t N)
{
  int glx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int grid_stride_x = (gridDim.x * blockDim.x);
  int r = (K - 1) / 2;

  for (int gx = glx; gx < N; gx += grid_stride_x)
  {
    float sum = 0.0f;
    for (int h = 0; h < K; ++h)
    {
      int a_idx = gx + h - r;
      if (0 <= a_idx && a_idx < N)
        sum += gm_a[a_idx] * g_const_b[h];
    }
    gm_c[gx] = sum;
  }
}

__global__ void device_1d_conv_tiled(
  const float *gm_a,
  const float *gm_b,
  float *gm_c,
  size_t K,
  size_t N)
{

  const int R = (K - 1) / 2;
  extern __shared__ float sm_a[];
  int total_blocks_x = (N + blockDim.x - 1) / blockDim.x;

  //grid stride at block level
  for (int bx = blockIdx.x; bx < total_blocks_x; bx += gridDim.x)
  {
    //load data from global to shared
    int gx = (bx * blockDim.x) + threadIdx.x;

    //load left halo
    // block_dim = 256, R = 2, smem_size = 256 + (R * 2) = 256 + 4 = 260
    // left halo: sm[0]..sm[1] || main elements: sm[2]..sm[257] || right halo: sm[258]..sm[259]
    if (threadIdx.x == 0 ) 
    {
      //load all left padding
      for (int i = 0; i < R; ++i) 
      {
        int load_idx = gx + i - R;
        //for cross block load load actual else clear memory
        sm_a[i] = (0 <= load_idx && load_idx < N) ? gm_a[load_idx] : 0.0f;
      }
    }

    //load own position
    sm_a[threadIdx.x + R] = (gx < N) ? gm_a[gx] : 0.0f;

    //load right halo
    if (threadIdx.x == blockDim.x - 1)
    {
      for (int i = 0; i < R; ++i)
      {
        int load_idx = gx + 1 + i;
        sm_a[blockDim.x + R + i] = (load_idx < N)  ? gm_a[load_idx] : 0.0f; 
      }
    }

    //wait for tile to finish writting
    __syncthreads();

    if (gx < N)
    {
      float sum = 0.0f;
      
      for ( int h = 0; h < K; ++h)
        sum += sm_a[threadIdx.x + h] * gm_b[h];

      gm_c[gx] = sum;
    }
    __syncthreads(); 
  }
}

__global__ void device_1d_conv_tiled_const(
  const float *gm_a,
  float *gm_c,
  size_t K,
  size_t N)
{

  const int R = (K - 1) / 2;
  extern __shared__ float sm_a[];
  int total_blocks_x = (N + blockDim.x - 1) / blockDim.x;

  for (int bx = blockIdx.x; bx < total_blocks_x; bx += gridDim.x)
  {
    int gx = (bx * blockDim.x) + threadIdx.x;

    if (threadIdx.x == 0 )
    {
      for (int i = 0; i < R; ++i)
      {
        int load_idx = gx + i - R;
        sm_a[i] = (0 <= load_idx && load_idx < N) ? gm_a[load_idx] : 0.0f;
      }
    }

    sm_a[threadIdx.x + R] = (gx < N) ? gm_a[gx] : 0.0f;

    if (threadIdx.x == blockDim.x - 1)
    {
      for (int i = 0; i < R; ++i)
      {
        int load_idx = gx + 1 + i;
        sm_a[blockDim.x + R + i] = (load_idx < N)  ? gm_a[load_idx] : 0.0f;
      }
    }

    __syncthreads();

    if (gx < N)
    {
      float sum = 0.0f;

      for ( int h = 0; h < K; ++h)
        sum += sm_a[threadIdx.x + h] * g_const_b[h];

      gm_c[gx] = sum;
    }
    __syncthreads();
  }
}

__global__ void device_1d_conv_tiled_block_stride(
  const float *gm_a,
  const float *gm_b,
  float *gm_c,
  size_t K,
  size_t N)
{
  const int R = (K - 1) / 2;
  extern __shared__ float sm_a[];
  const int total_blocks_x = (N + blockDim.x - 1) / blockDim.x;
  const int total_elements = blockDim.x + (R * 2);

  //grid stride loop at block level
  for (int bx = blockIdx.x; bx < total_blocks_x; bx += gridDim.x)
  {
    //load tile
    for (int lx = threadIdx.x; lx < total_elements; lx += blockDim.x)
    {
      //normally we have 256 blockdim.x
      //
      //left halo = sm[0..1] ==> sm[0...R-1]
      //center = sm[2...257] ==> sm[R...blockDim.x + R - 1]
      //right halo = sm[258..259] => sm[blockDim.x + R ...blockDim.x + R + 1]
      int gx = (bx * blockDim.x) + lx;
      int load_idx = gx - R;
      sm_a[lx] = (0 <= load_idx && load_idx < N) ? gm_a[load_idx] : 0.0f; 
    }

    __syncthreads();

    //convolution
    float rsum = 0.0f;
    for (int h = 0; h < K; ++h)
      rsum += sm_a[threadIdx.x + h] * gm_b[h];
    
    int glx = (bx * blockDim.x) + threadIdx.x;
    if (glx < N)
      gm_c[glx] = rsum;
    
    __syncthreads();
  }
}

__global__ void device_1d_conv_tiled_block_stride_const(
  const float *gm_a,
  float *gm_c,
  size_t K,
  size_t N)
{
  const int R = (K - 1) / 2;
  extern __shared__ float sm_a[];
  const int total_blocks_x = (N + blockDim.x - 1) / blockDim.x;
  const int total_elements = blockDim.x + (R * 2);

  for (int bx = blockIdx.x; bx < total_blocks_x; bx += gridDim.x)
  {
    for (int lx = threadIdx.x; lx < total_elements; lx += blockDim.x)
    {
      int gx = (bx * blockDim.x) + lx;
      int load_idx = gx - R;
      sm_a[lx] = (0 <= load_idx && load_idx < N) ? gm_a[load_idx] : 0.0f;
    }

    __syncthreads();

    float rsum = 0.0f;
    for (int h = 0; h < K; ++h)
      rsum += sm_a[threadIdx.x + h] * g_const_b[h];

    int glx = (bx * blockDim.x) + threadIdx.x;
    if (glx < N)
      gm_c[glx] = rsum;

    __syncthreads();
  }
}

// 1D convolution with zero padding and a centered kernel (cross-correlation).
//
// Let r = (K - 1) / 2. Out-of-bounds accesses to A are treated as zero.
// C[i] = sum_{j=0..K-1} A[i + j - r] * B[j]
//
// The kernel slides over the input signal, computing the sum of
// element-wise multiplications at each position. Zero padding is used
// at the boundaries where the kernel extends beyond the input signal.
//
// Input:
// - A: vector of size N (input signal)
// - B: vector of size K (convolution kernel)
//
// Output:
// - C: vector of size N (convolved signal)
//
// Notes:
// - K is odd and smaller than N
// - Output size is N (same as input) due to padding
// - Matches PyTorch torch.nn.functional.conv1d(..., padding=K//2)
//   (cross-correlation, kernel is not flipped)
// - Adapted from KernelBench
void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
  cudaEvent_t total_start = nullptr;
  cudaEvent_t total_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  CUDA_CHECK(cudaEventRecord(total_start));

  const size_t r = (K > 0) ? (K - 1) / 2 : 0;
  const size_t outN = N;
  const bool use_constant_filter = kernel_uses_constant_filter(g_kernel_kind);
  if (use_constant_filter && K > kMaxConstantFilterSize) {
    std::cerr << "Filter size " << K
              << " exceeds constant-memory comparison limit "
              << kMaxConstantFilterSize << '\n';
    return;
  }

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, outN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, A, N * sizeof(float),cudaMemcpyHostToDevice));
  if (use_constant_filter) {
    CUDA_CHECK(cudaMemcpyToSymbol(g_const_b, B, K * sizeof(float)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_B, K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * sizeof(float), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemset(d_C, 0, outN * sizeof(float)));

  dim3 block_shape(g_launch_config.block_x, 1);
  dim3 grid_shape(g_launch_config.grid_x, 1);

  CUDA_CHECK(cudaEventRecord(kernel_start));
  switch (g_kernel_kind) {
    case KernelKind::Basic:
      device_1d_conv_basic<<<grid_shape, block_shape>>>(d_A, d_B, d_C, K, N);
      break;
    case KernelKind::BasicConst:
      device_1d_conv_basic_const<<<grid_shape, block_shape>>>(d_A, d_C, K, N);
      break;
    case KernelKind::Tiled:
    {
      size_t smem_bytes = (block_shape.x + (r * 2)) * sizeof(float);
      device_1d_conv_tiled<<<grid_shape, block_shape, smem_bytes>>>(d_A, d_B, d_C, K, N);
      break;
    }
    case KernelKind::TiledConst:
    {
      size_t smem_bytes = (block_shape.x + (r * 2)) * sizeof(float);
      device_1d_conv_tiled_const<<<grid_shape, block_shape, smem_bytes>>>(d_A, d_C, K, N);
      break;
    }
    case KernelKind::BlockStrideStub:
    {
      size_t smem_bytes = (block_shape.x + (r * 2)) * sizeof(float);
      device_1d_conv_tiled_block_stride<<<grid_shape, block_shape, smem_bytes>>>(
        d_A, d_B, d_C, K, N);
      break;
    }
    case KernelKind::BlockStrideConst:
    {
      size_t smem_bytes = (block_shape.x + (r * 2)) * sizeof(float);
      device_1d_conv_tiled_block_stride_const<<<grid_shape, block_shape, smem_bytes>>>(
        d_A, d_C, K, N);
      break;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(kernel_stop));

  CUDA_CHECK(cudaMemcpy(C, d_C, outN * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));

  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
  g_last_timing.total_ms = total_ms;
  g_last_timing.kernel_ms = kernel_ms;
  if (g_print_timing) {
    std::cerr << "total_ms=" << total_ms << " kernel_ms=" << kernel_ms << '\n';
  }

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  bool skip_cpu_verify = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--skip-cpu") {
      skip_cpu_verify = true;
    } else if (!parse_kernel_arg(argv[i])) {
      return 1;
    }
  }
  return run_tests(skip_cpu_verify);
}
