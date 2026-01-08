#include<cstdio>
#include<cuda_runtime.h>
#include<vector>
/**
    vector 的参数 是 std::vector<T, std::allocator<T>>
    std::allocator 是c++标准库中提供的内存分配器

    allocator的接口:
    T *allocate(size_t n) 分配n个T类型的内存
    void deallocate(T* p, size_t n) 释放p指向的n个T类型的内存

    自定义allocator直接实现了在GPU上进行内存分配
*/
template<typename T>
class CudaAllocator{
public:
    using value_type = T;
    T* allocate(size_t n){
        T* p = nullptr;
        cudaMallocManaged(&p, n*sizeof(T));
        return p;
    }

    void deallocate(T* p, size_t n){
        cudaFree(p);
    }
};

__global__ void kernel(int* arr, int n){
    // 网格跨步循环
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x){
        arr[i] = i;
    }
}
int main(){
    int n = 65535;
    auto vec = std::vector<int, CudaAllocator<int>>(n);
    kernel<<<32, 256>>>(vec.data(), n);
    cudaDeviceSynchronize();

    for(int i = 0; i < 10; i++){
        printf("%d ", vec[i]);
    }

}