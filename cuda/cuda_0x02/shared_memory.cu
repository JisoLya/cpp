#include"..\cuda_allocator.cuh"

__global__ void parallel_sum(int* sum, int* data, int n){
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < n / 1024;
    i += blockDim.x * gridDim.x){
        int local_sum = 0;
        for(int j = i * 1024; j < i * 1024 + 1024; ++j){
            local_sum += data[j];
        }

        sum[i] = local_sum;
    }
}

int main(){
    int n = 65536 * 2;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> result(n/1024);

    for(int i = 0; i < n;++i){
        arr[i] = 1;
    }

    // 需要注意的是 <<<blocks, threads>>>这样声明出来的总线程数是blocks * threads
    // 不过只有同一个block内部的线程是共享内存的
    parallel_sum<<<n / 1024 / 128, 128>>>(result.data(), arr.data(), n);
    cudaDeviceSynchronize();

    int final_sum = 0;

    for(int i = 0; i < n / 1024; ++i){
        final_sum += result[i];
    }

    printf("Final sum: %d\n", final_sum);
}