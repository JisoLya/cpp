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

/**
    实现标量的 aX + Y操作
*/
template<typename Func>
__global__ void parallel_for(int n, Func f){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        f(i);
    }
}

int main(){
    int n = 65535;
    std::vector<float, CudaAllocator<float>> X(n);
    std::vector<float, CudaAllocator<float>> Y(n);

    float a = 2.0f;
    for(int i = 0;i < n;++i){
        X[i] = std::rand() * (1.0f / RAND_MAX);
        Y[i] = std::rand() * (1.0f / RAND_MAX);
    }

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    parallel_for<<<numBlocks, blockSize>>>(n, [a, x_data = X.data(), y_data = Y.data()] __device__ (int i){
        x_data[i] = a * x_data[i] + y_data[i];
    });


    cudaDeviceSynchronize();

    for(int i = 0;i < n; ++i){
        printf("X[%d] = %f\n", i, X[i]);
    }
}   