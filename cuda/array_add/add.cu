#include<stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>

/**
    global_id = threadIdx.x + blockIdx.x * blockDim.x
    total thread number = gridDim.x * blockDim.x
*/

void vector_add_cpu(const float* a, const float* b, float* c, int n){
    for(int i=0; i<n; ++i){
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(const float* a, const float* b, float* c, int n){
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id < n){
        c[global_id] = a[global_id] + b[global_id];
    }
}

void init_data(std::vector<float>& data){
    // 1. 定义随机数引擎（使用种子初始化）
    std::random_device rd;  // 用于获取真实的随机种子
    std::mt19937 gen(rd()); // 经典的梅森旋转算法引擎

    // 2. 定义分布范围 [min, max]
    std::uniform_int_distribution<> dis(1, 100);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<float>(dis(gen)); });
}


int main(){
    // 100万个float元素相加
    const int N = 100000000;
    size_t size = N * sizeof(float);

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    init_data(h_a);
    init_data(h_b);


    // --- CPU 测试 ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vector_add_cpu(h_a.data(), h_b.data(), h_c.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    printf("Element size = %zu bytes, CPU vector add time: %f ms\n", size, cpu_time.count());


    // GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);


    // auto start_gpu = std::chrono::high_resolution_clock::now();
    /**
        如果在这里计时， 内存拷贝的时间会占用很大一部分..导致看起来kernel的运行时长很长， 其实大部分时间都花在内存拷贝上了
        如果只看计算时长，那么时间是远小于的CPU的
    */
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);
    //经验值 block size 256
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vector_add_gpu<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    std::chrono::duration<double, std::milli> kernel_time = std::chrono::high_resolution_clock::now() - start_gpu;
    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    printf("Element size = %zu bytes, GPU vector add time: %f ms\n", size, gpu_time.count());

    // 打印对比结果
    std::cout << "--- 数组规模: " << N << " ---" << std::endl;
    std::cout << "CPU 耗时: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU 核函数耗时: " << kernel_time.count() << " ms" << std::endl;
    std::cout << "GPU 耗时 (含内存传输): " << gpu_time.count() << " ms" << std::endl;

    // 清理显存
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}