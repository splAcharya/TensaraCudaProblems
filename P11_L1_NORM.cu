#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Problem 11: L1 Normalization
 * Source: https://tensara.org/problems/l1-norm
 *
 * Normalize each row of a two-dimensional tensor by the sum of the
 * absolute values in that row.
 *
 *   Y[b][d] = X[b][d] / (sum(abs(X[b][d])) + epsilon)
 *
 * The reduction is over the second dimension D. The input and output have
 * the same row-major shape (B, D).
 *
 * Published shapes:
 * - (128, 4096)
 * - (256, 4096)
 * - (128, 8192)
 * - (256, 8192)
 * - (128, 16384)
 */

// Tensara-style signature:
// - input and output are device pointers
// - B is the row count and D is the feature count
extern "C" void solution(const float* input, float* output, size_t B,
                         size_t D);

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

enum class KernelVariant {
  kBasic,
  kFloat4,
  kShared,
  kSharedFloat4,
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
    case KernelVariant::kFloat4:
      return "float4";
    case KernelVariant::kShared:
      return "shared";
    case KernelVariant::kSharedFloat4:
      return "shared_float4";
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
  if (value == "shared") {
    g_kernel_variant = KernelVariant::kShared;
    g_kernel_arg_set = true;
    return true;
  }
  if (value == "shared_float4") {
    g_kernel_variant = KernelVariant::kSharedFloat4;
    g_kernel_arg_set = true;
    return true;
  }

  std::cerr << "Unknown kernel: " << value
            << " (use --kernel=basic, --kernel=float4, or "
            << "--kernel=shared, or --kernel=shared_float4)\n";
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
  size_t rows = 0;
  size_t cols = 0;
  std::vector<float> input;
  std::vector<float> expected;
};

// CPU reference stub. Keep this body empty until the reference is requested.
static void cpu_l1_norm_reference(const std::vector<float>& input,
                                  std::vector<float>& output, size_t rows,
                                  size_t cols) {
  
  for (size_t r = 0; r < rows; ++r)
  {
    //get current column sum
    float col_sum = 0.0f;
    for (size_t c = 0; c < cols; ++c)
      col_sum += std::fabs(input[r * cols + c]);
      
    //perform normalization
    for (size_t c = 0; c < cols; ++c)
      output[r * cols + c] = input[r * cols + c] / col_sum;
  }
}

// Basic GPU kernel stub. Keep this body empty until the kernel is requested.
__global__ void l1_norm_basic_kernel(const float* input, float* output,
                                     size_t rows, size_t cols) {
  
  size_t gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t total = rows * cols;
  size_t grid_stride = (blockDim.x * gridDim.x);

  for (size_t gx = gid; gx < total; gx += grid_stride)
  {
    size_t row_pos = gx / cols;
    //sum entire column
    float col_sum = 0.0f;
    for (size_t r = 0; r < cols; ++r)
      col_sum += std::fabs(input[row_pos * cols + r]);

    //update output
    output[gx] = input[gx] / col_sum;
  }
}

// Float4 GPU kernel with scalar prefix and tail handling.
__global__ void l1_norm_basic_float4_kernel(const float* input,
                                            float* output, size_t rows,
                                            size_t cols) {

  size_t gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t total = rows * cols;
  size_t grid_stride = (blockDim.x * gridDim.x);
  for (size_t gx = gid; gx < total; gx += grid_stride)
  {
    //2D index
    size_t row_pos = gx / cols;
    size_t col_pos = gx % cols;
    
    /* Scalar Prefix
       0  1  2  3  4   5
       6  7  8  9  10  11

       ----4-----  ----4-----  ----4------
       0  1  2  3  4  5  6  7  8  9  10  11       
                   0  1  2  3
       take 6
       position in float4: (1 * 6) % 4 ==> 6 % 4 = 2
       see above in the second float4, 6 is the 3rd item(2nd 0indexed)

       prefix that start at non divisible position
       6 7 
       how many prefx count = (4 - f4 pos) % 4 = (4 - 2) % 4 = 2%4 =2
       which macthed 2 prefxies 6 and 7
    */
    size_t pos_in_f4 = (row_pos * cols) % 4;
    size_t prefix_count = (4 - pos_in_f4) % 4;

    float abs_sum = 0.0f;
    for (size_t i = 0; i < prefix_count; ++i)
    {
      size_t flat_idx = row_pos * cols + i;
      abs_sum += std::fabs(input[flat_idx]);
    }

    //float4 chunks
    size_t remain = cols - prefix_count;
    size_t f4_count = remain / 4;
    size_t vector_start = (row_pos * cols) + prefix_count;
    const float4* input4 = reinterpret_cast<const float4*>(
        &(input[vector_start]));
    for (size_t i = 0; i < f4_count; ++i)
    {
      const float4 temp4 = input4[i];
      abs_sum += std::fabs(temp4.x);
      abs_sum += std::fabs(temp4.y);
      abs_sum += std::fabs(temp4.z);
      abs_sum += std::fabs(temp4.w);     
    }

    //scalar tail
    size_t tail_start = prefix_count + (f4_count * 4);
    for (size_t i = tail_start; i < cols; ++i)
    {
      abs_sum += std::fabs(input[row_pos * cols + i]);
    }
    /*--------------INPUT READ PART DONE----------------*/

    /*
      0 1 2 3 4  5
      6 7 8 9 10 11 
      
      ---4--- ---4--- ----4----
      0 1 2 3 4 5 6 7 8 9 10 11

      take 1, 2
      prefix_count = 0
      col_pos = 1 % 6 = 1, 2 % 6 = 2
      (1 - 0) % 4 =  1
      (2 - 0) % 4 =  2, 
      both not a float 4 leaders

      take 8, 9
      prefix_count = 2
      col_pos = 8 % 6 = 2, 9 % 6 = 3
      (2 - 2) % 4 = 0, not a f4 leader
      (3 - 2) % 4 = 1, not a f4 lader 
    */

    bool in_f4_range = (prefix_count <= col_pos) && (col_pos < tail_start);
    bool is_f4_leader = in_f4_range && ( ((col_pos - prefix_count) % 4) == 0);

    //prefix
    if (col_pos < prefix_count)
    {
      output[gx] = input[gx] / abs_sum;
    }
    //vector
    else if (is_f4_leader)
    {
      const float4 *input4 = reinterpret_cast<const float4 *>(&(input[gx]));
      float4 *output4 = reinterpret_cast<float4 *>(&(output[gx]));
      output4[0].x = input4[0].x / abs_sum;
      output4[0].y = input4[0].y / abs_sum;
      output4[0].z = input4[0].z / abs_sum;
      output4[0].w = input4[0].w / abs_sum;
    }
    //tail
    else if (col_pos >= tail_start)
    {
      output[gx] = input[gx] / abs_sum;
    }
  }
}

// Shared-memory GPU kernel.
__global__ void l1_norm_shared_kernel(const float* input, float* output,
                                      size_t rows, size_t cols) {
  extern  __shared__ float smem_ar[];

  //each block owns an output row
  for (size_t bx = blockIdx.x; bx < rows; bx += gridDim.x)
  {
    //clear shared_memory
    smem_ar[threadIdx.x] = 0.0f;
    
    //cooperative partial sum, no need to sync since,
    //every thread is just writting on its own portion
    for (size_t lx = threadIdx.x; lx < cols; lx += blockDim.x)
      smem_ar[threadIdx.x] += std::fabs(input[bx * cols + lx]); 

    //wait for the full column to be loaded into smem
    __syncthreads();

    //reduction
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
      if (threadIdx.x < offset)
        smem_ar[threadIdx.x] += smem_ar[threadIdx.x + offset];
      
      //wait to read update value 
      __syncthreads();
    }

    // 1st index in smem_ar holds the final reduction, 
    //load it into local reg for fast access,
    //should also result in broadcast as well
    const float abs_sum = smem_ar[0]; 

    //coperatively update entire output column
    for (size_t lx = threadIdx.x; lx < cols; lx += blockDim.x)
      output[bx * cols + lx] = input[bx * cols + lx] / abs_sum;
  }
}

// Shared-memory float4 GPU kernel stub.
__global__ void l1_norm_shared_float4_kernel(const float* input,
                                             float* output, size_t rows,
                                             size_t cols) {
  extern __shared__ float smem_ar[];
  
  for (size_t bx = blockIdx.x; bx < rows; bx += gridDim.x)
  {
    /* clear shared memory */
    smem_ar[threadIdx.x] = 0.0f;

    /* scalar prefix
      give starting row_idx, in a group of float 4 chunks
      where does the starting row_idx fall in ? 
    */
    size_t pos_in_f4 = (bx * cols) % 4;
    size_t prefix_count = (4 - pos_in_f4) % 4;
    if (prefix_count > cols)
      prefix_count = cols;

    //coperatively load scalar prefix
    for (size_t lx = threadIdx.x; lx < prefix_count; lx += blockDim.x)
      smem_ar[threadIdx.x] += std::fabs(input[bx * cols + lx]);

    size_t remain   = cols - prefix_count;
    size_t f4_count = remain / 4;
    size_t f4_start = (bx * cols) + prefix_count;
    const float4 *input4 = reinterpret_cast<const float4 *>(&input[f4_start]);
    for (size_t lx = threadIdx.x; lx < f4_count; lx += blockDim.x)
    {
      const float4 temp4 = input4[lx];
      float temp = 0.0f;
      temp += std::fabs(temp4.x);
      temp += std::fabs(temp4.y);
      temp += std::fabs(temp4.z);
      temp += std::fabs(temp4.w);
      smem_ar[threadIdx.x] += temp;
    }

    //cooperatively load scalar tail
    size_t tail_start = prefix_count + (f4_count * 4);
    for (size_t lx = threadIdx.x + tail_start; lx < cols; lx += blockDim.x)
      smem_ar[threadIdx.x] += std::fabs(input[bx * cols + lx]);

    //wait for partial sums to complete
    __syncthreads();

    //reduce
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
      if (threadIdx.x < offset)
        smem_ar[threadIdx.x] += smem_ar[threadIdx.x + offset];
      
      __syncthreads();
    }

    //load reduceiotnj result into local reg
    float abs_sum = smem_ar[0];

    //update output
    //cooperatively update scalar prefix cols
    for (size_t lx = threadIdx.x; lx < prefix_count; lx += blockDim.x)
      output[bx * cols + lx] = input[bx * cols + lx] / abs_sum;

    //cooperatively update float4 chunks
    float4 *output4 = reinterpret_cast<float4 *>(&(output[f4_start]));
    for (size_t lx = threadIdx.x; lx < f4_count; lx += blockDim.x)
    {
      float4 temp4 = input4[lx];
      temp4.x = temp4.x / abs_sum;
      temp4.y = temp4.y / abs_sum;
      temp4.z = temp4.z / abs_sum;
      temp4.w = temp4.w / abs_sum;
      output4[lx] = temp4;
    }

    //coperatively update scalar tail
    for (size_t lx = threadIdx.x + tail_start; lx < cols; lx +=  blockDim.x)
      output[bx * cols + lx] = input[bx * cols + lx] / abs_sum;
  }
}

__global__ void l1_norm_warp_kernel(const float* input, float* output,
                                    size_t rows, size_t cols) 
{

}

extern "C" void solution(const float* input, float* output, size_t B,
                         size_t D) {
  const dim3 block_shape(g_launch_config.block_x, 1, 1);
  const dim3 grid_shape(g_launch_config.grid_x, 1, 1);

  switch (g_kernel_variant) {
    case KernelVariant::kBasic:
      l1_norm_basic_kernel<<<grid_shape, block_shape>>>(input, output, B, D);
      break;
    case KernelVariant::kFloat4:
      l1_norm_basic_float4_kernel<<<grid_shape, block_shape>>>(
          input, output, B, D);
      break;
    case KernelVariant::kShared: {
      const size_t shared_bytes =
          static_cast<size_t>(g_launch_config.block_x) * sizeof(float);
      l1_norm_shared_kernel<<<grid_shape, block_shape, shared_bytes>>>(
          input, output, B, D);
      break;
    }
    case KernelVariant::kSharedFloat4: {
      const size_t shared_bytes =
          static_cast<size_t>(g_launch_config.block_x) * sizeof(float);
      l1_norm_shared_float4_kernel<<<grid_shape, block_shape, shared_bytes>>>(
          input, output, B, D);
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

static std::vector<float> make_l1_input(size_t rows, size_t cols) {
  const size_t total = rows * cols;
  std::vector<float> input(total, 0.0f);
  for (size_t i = 0; i < total; ++i) {
    const int raw =
        static_cast<int>((i * 37 + rows * 11 + cols * 7 + 19) % 257) - 128;
    input[i] = static_cast<float>(raw) / 32.0f;
  }
  return input;
}

static bool verify_close(const std::vector<float>& got,
                         const std::vector<float>& expected, float atol,
                         float rtol, const char* label, bool verbose) {
  if (got.size() != expected.size()) {
    if (verbose) {
      std::cerr << "verify(" << label << "): size mismatch got="
                << got.size() << " expected=" << expected.size() << '\n';
    }
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

  if (!ok && verbose) {
    std::cerr << "verify(" << label << "): FAIL at i=" << first_bad
              << " got=" << got[first_bad]
              << " expected=" << expected[first_bad]
              << " max_abs=" << max_abs << " max_i=" << max_i << '\n';
  }
  return ok;
}

static std::vector<float> run_gpu_case(const TestCase& test) {
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t elements = test.rows * test.cols;
  const size_t bytes = elements * sizeof(float);
  std::vector<float> output(elements, 0.0f);

  CUDA_CHECK(cudaMalloc(&device_input, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_input, test.input.data(), bytes,
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < kTimingWarmupIterations; ++i) {
    solution(device_input, device_output, test.rows, test.cols);
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
    solution(device_input, device_output, test.rows, test.cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_samples.push_back(elapsed_ms);
    kernel_samples.push_back(elapsed_ms);
  }

  g_last_timing.total_ms = select_timing_sample(total_samples);
  g_last_timing.kernel_ms = select_timing_sample(kernel_samples);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return output;
}

struct TestResult {
  std::string group;
  std::string name;
  std::string kernel;
  size_t rows = 0;
  size_t cols = 0;
  size_t elements = 0;
  int block_x = 0;
  int grid_x = 0;
  std::string cpu;
  std::string gpu;
  float total_ms = 0.0f;
  float kernel_ms = 0.0f;
};

static void print_results_table(const std::vector<TestResult>& results) {
  std::cout << std::left << std::setw(10) << "group" << std::setw(24)
            << "name" << std::setw(14) << "kernel" << std::setw(8)
            << "rows" << std::setw(8) << "cols" << std::setw(14)
            << "elements" << std::setw(8) << "block_x" << std::setw(8)
            << "grid_x" << std::setw(8) << "cpu" << std::setw(8)
            << "gpu" << std::setw(12) << "total_ms" << std::setw(12)
            << "kernel_ms" << '\n';
  std::cout << std::string(138, '-') << '\n';
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& result : results) {
    std::cout << std::left << std::setw(10) << result.group
              << std::setw(24) << result.name << std::setw(14)
              << result.kernel << std::setw(8) << result.rows << std::setw(8)
              << result.cols << std::setw(14) << result.elements
              << std::setw(8) << result.block_x << std::setw(8)
              << result.grid_x << std::setw(8) << result.cpu << std::setw(8)
              << result.gpu << std::setw(12) << result.total_ms
              << std::setw(12) << result.kernel_ms << '\n';
  }
}

static void print_scale_summary(const std::vector<TestResult>& results) {
  std::vector<std::pair<std::string, std::string>> keys;
  for (const auto& result : results) {
    if (result.group != "scale") {
      continue;
    }
    const std::pair<std::string, std::string> key(result.name, result.kernel);
    if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
      keys.push_back(key);
    }
  }

  if (keys.empty()) {
    return;
  }

  std::cout << "\nScaling summary (" << timing_mode_name()
            << " kernel_ms, lower is better)\n";
  for (const auto& key : keys) {
    const TestResult* best = nullptr;
    for (const auto& result : results) {
      if (result.group == "scale" && result.name == key.first &&
          result.kernel == key.second &&
          (best == nullptr || result.kernel_ms < best->kernel_ms)) {
        best = &result;
      }
    }
    if (best != nullptr) {
      std::cout << "  " << key.second << " / " << key.first << ": "
                << best->kernel_ms << " ms at (" << best->block_x << ", "
                << best->grid_x << ")\n";
    }
  }
}

static int run_profile() {
  const TestCase test{
      "profile", 128, 4096, make_l1_input(128, 4096), {}};
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t bytes = test.input.size() * sizeof(float);
  CUDA_CHECK(cudaMalloc(&device_input, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_input, test.input.data(), bytes,
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < kProfileWarmupIterations; ++i) {
    solution(device_input, device_output, test.rows, test.cols);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kProfileIterations; ++i) {
    solution(device_input, device_output, test.rows, test.cols);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  std::cout << "profile scope=kernel-only verify=off"
            << " kernel=" << current_kernel_name()
            << " rows=" << test.rows << " cols=" << test.cols
            << " block_x=" << g_launch_config.block_x
            << " grid_x=" << g_launch_config.grid_x
            << " warmup=" << kProfileWarmupIterations
            << " repeats=" << kProfileIterations
            << " avg_kernel_ms=" << elapsed_ms / kProfileIterations << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return 0;
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
      {"small_2x3", 2, 3,
       {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f},
       {1.0f / 6.0f, -2.0f / 6.0f, 3.0f / 6.0f,
        -4.0f / 15.0f, 5.0f / 15.0f, -6.0f / 15.0f}},
      {"small_2x4", 2, 4,
       {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 2.0f, -3.0f, 4.0f},
       {0.25f, 0.25f, 0.25f, 0.25f, -0.1f, 0.2f, -0.3f, 0.4f}},
      {"single_element", 1, 1, {-3.0f}, {-1.0f}},
  };
  const std::vector<TestCase> medium_tests = {
      {"medium_3x17", 3, 17, {}, {}},
      {"medium_7x65", 7, 65, {}, {}},
      {"medium_31x257", 31, 257, {}, {}},
  };
  const std::vector<TestCase> large_tests = {
      {"large_64x1024", 64, 1024, {}, {}},
      {"large_257x4097", 257, 4097, {}, {}},
  };
  const std::vector<TestCase> published_tests = {
      {"tensara_1", 128, 4096, {}, {}},
      {"tensara_2", 256, 4096, {}, {}},
      {"tensara_3", 128, 8192, {}, {}},
      {"tensara_4", 256, 8192, {}, {}},
      {"tensara_5", 128, 16384, {}, {}},
  };
  const std::vector<TestCase> shape_tests = {
      {"shape_3x33", 3, 33, {}, {}},
      {"shape_65x129", 65, 129, {}, {}},
      {"shape_127x513", 127, 513, {}, {}},
  };
  const std::vector<TestCase> tail_tests = {
      {"tail_5x257", 5, 257, {}, {}},
      {"tail_17x1025", 17, 1025, {}, {}},
      {"tail_33x4099", 33, 4099, {}, {}},
  };
  const int scale_block_sizes[] = {64, 128, 256, 512};
  const int scale_grid_sizes[] = {8, 16, 32, 64, 128};
  const KernelVariant kernel_variants[] = {KernelVariant::kBasic,
                                           KernelVariant::kFloat4,
                                           KernelVariant::kShared,
                                           KernelVariant::kSharedFloat4};

  std::vector<TestResult> results;
  bool all_ok = true;

  auto run_case = [&](const char* group, const TestCase& test,
                      const std::vector<float>& input,
                      const std::vector<float>* expected) {
    TestResult result;
    result.group = group;
    result.name = test.name;
    result.kernel = current_kernel_name();
    result.rows = test.rows;
    result.cols = test.cols;
    result.elements = input.size();
    result.block_x = g_launch_config.block_x;
    result.grid_x = g_launch_config.grid_x;
    result.cpu = "SKIP";
    result.gpu = "SKIP";
    g_last_timing = {};

    std::vector<float> reference;
    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      reference.assign(input.size(), 0.0f);
      cpu_l1_norm_reference(input, reference, test.rows, test.cols);
      result.cpu = "REF";
    }

    if (kGpuKernelImplemented) {
      TestCase gpu_test = test;
      gpu_test.input = input;
      const std::vector<float> output = run_gpu_case(gpu_test);
      if (expected != nullptr) {
        const bool ok = verify_close(output, *expected, 1e-5f, 1e-5f,
                                     test.name, true);
        result.gpu = ok ? "PASS" : "FAIL";
        all_ok &= ok;
      } else if (!skip_cpu_verify && kCpuReferenceImplemented) {
        const bool ok = verify_close(output, reference, 1e-5f, 1e-5f,
                                     test.name, true);
        result.gpu = ok ? "PASS" : "FAIL";
        all_ok &= ok;
      }
    }

    result.total_ms = g_last_timing.total_ms;
    result.kernel_ms = g_last_timing.kernel_ms;
    results.push_back(result);
  };

  auto run_sized = [&](const char* group, const TestCase& test) {
    g_launch_config = default_launch;
    run_case(group, test, make_l1_input(test.rows, test.cols), nullptr);
  };

  auto run_scale = [&](const TestCase& test) {
    const std::vector<float> input = make_l1_input(test.rows, test.cols);
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
      g_launch_config = default_launch;
      run_case("small", test, test.input, &test.expected);
    }

    if (!skip_cpu_verify && kCpuReferenceImplemented) {
      for (const auto& test : medium_tests) {
        run_sized("medium", test);
      }
      for (const auto& test : large_tests) {
        run_sized("large", test);
      }
      for (const auto& test : shape_tests) {
        run_sized("shape", test);
      }
      for (const auto& test : tail_tests) {
        run_sized("tail", test);
      }
    }

    if (skip_cpu_verify) {
      for (const auto& test : published_tests) {
        run_sized("tensara", test);
      }
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
  bool profile_mode = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--skip-cpu") {
      skip_cpu_verify = true;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [--skip-cpu] [--profile] [--kernel=NAME]"
                << " [--timing=median|best] [--timing-repeats=N]"
                << " [--list-kernels]\n";
      return 0;
    } else if (arg == "--list-kernels") {
      std::cout << "basic\nfloat4\nshared\nshared_float4\n";
      return 0;
    } else if (arg == "--profile") {
      profile_mode = true;
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
      std::cerr << "Unknown argument: " << arg
                << " (supported: --skip-cpu, --profile, --kernel=..., "
                << "--timing=..., --timing-repeats=..., --list-kernels)\n";
      return 1;
    }
  }

  if (profile_mode) {
    if (!g_kernel_arg_set) {
      std::cerr << "--profile requires an explicit --kernel=... value\n";
      return 1;
    }
    if (kGpuKernelImplemented && !cuda_runtime_ready()) {
      return 1;
    }
    return run_profile();
  }
  return run_tests(skip_cpu_verify);
}
