#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

// 方案 1: 在循环内直接使用 atomicAdd (极其低效)
__global__ void sum_atomic_in_loop(int* sum, int const* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&sum[0], arr[i]); 
    }
}

// 方案 2: 每个线程先寄存器累加，最后一次 atomicAdd (你当前的代码)
__global__ void sum_atomic_after_loop(int* sum, int const* arr, int n) {
    int local_sum = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }
    atomicAdd(&sum[0], local_sum);
}

// 方案 3: 使用共享内存进行块内规约 (工业级标准)
__global__ void sum_shared_memory(int* sum, int const* arr, int n) {
    // 动态分配共享内存
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_sum = 0;

    // 1. 跨步循环累加到寄存器
    for (; i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // 2. 块内折半规约 (Tree Reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 每个 Block 只有一个线程去更新全局变量
    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

// 简单的计时包装器
struct Timer {
    cudaEvent_t start, stop;
    Timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    void begin() { cudaEventRecord(start); }
    float end() {
        float ms = 0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

int main() {
    const int n = 1 << 24; // 约1600万个数据，让差距更明显
    const size_t bytes = n * sizeof(int);

    int *d_arr, *d_sum;
    cudaMallocManaged(&d_arr, bytes);
    cudaMallocManaged(&d_sum, sizeof(int));

    // 初始化数据
    for (int i = 0; i < n; i++) d_arr[i] = 1;

    Timer timer;
    int threads = 256;
    int blocks = 256;

    // --- 测试方案 1 ---
    *d_sum = 0;
    timer.begin();
    sum_atomic_in_loop<<<blocks, threads>>>(d_sum, d_arr, n);
    printf("方案1 (循环内原子加) 时间: %f ms, 结果: %d\n", timer.end(), *d_sum);

    // --- 测试方案 2 ---
    *d_sum = 0;
    timer.begin();
    sum_atomic_after_loop<<<blocks, threads>>>(d_sum, d_arr, n);
    printf("方案2 (线程级原子加) 时间: %f ms, 结果: %d\n", timer.end(), *d_sum);

    // --- 测试方案 3 ---
    *d_sum = 0;
    timer.begin();
    // 共享内存大小为 threads * sizeof(int)
    sum_shared_memory<<<blocks, threads, threads * sizeof(int)>>>(d_sum, d_arr, n);
    printf("方案3 (共享内存规约) 时间: %f ms, 结果: %d\n", timer.end(), *d_sum);

    cudaFree(d_arr);
    cudaFree(d_sum);
    return 0;
}