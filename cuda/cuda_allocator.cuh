#include<cstdio>
#include<cuda_runtime.h>
#include<vector>

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