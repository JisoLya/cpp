#include<cstdio>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>

__global__ void parrallel_sum(int* sum, int const* arr, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // sum[0] += arr[i];
        atomicAdd(&sum[0], arr[i]);
    }
}

template<typename T>
class CudaAllocator {
public:
    using value_type = T;
    T* allocate(size_t n) {
        T* p = nullptr;
        cudaMallocManaged(&p, n * sizeof(T));
        return p;
    }

    void deallocate(T* p, size_t n) {
        cudaFree(p);
    }
};


int main() {
    int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(1);

    for (int i = 0; i < n; i++) {
        arr[i] = 1;
    }

    parrallel_sum <<<256, 256>>> (sum.data(), arr.data(), n);
    cudaDeviceSynchronize();

    /**
    这里输入的sum很小？ 为什么呢
    因为我们的GPU是并行计算的：
    其中这里的sum[0] += arr[i];
    会被解释成为 读取sum[0] 到寄存器A， 读取arr[i]到寄存器B， 寄存器A + 寄存器B， 然后把结果写回sum[0]
    但是多个线程同时进行这个操作时，会出现数据竞争，导致最终错误
    所以应该使用atomicAdd()保证原子性

    */
    printf("Sum: %d\n", sum[0]);
    return 0;
}