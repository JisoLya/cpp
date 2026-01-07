#include<stdio.h>
#include <cuda_runtime.h>

void hello_from_cpu() {
    printf("Hello from CPU!\n");
}

__global__ void hello_cuda(){
    printf("Hello from CUDA kernel!\n");
}

int main(){
    hello_from_cpu();
    hello_cuda<<<3,2>>>();
    // grid_size 3 网格中有多少个block
    // block_size 2 每个block中有多少个线程
    // 那么我该怎么获取到线程的全局id呢？
    /**
        global_id = threadIdx.x + blockIdx.x * blockDim.x
    */
    cudaDeviceSynchronize(); // 等待所有CUDA线程完成

    return 0;
}