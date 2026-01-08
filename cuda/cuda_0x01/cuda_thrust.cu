#include<cstdio>
#include<cuda_runtime.h>
#include<thrust/universal_vector.h>

template<typename Func>
__global__ void parallel_for(int n, Func f){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        f(i);
    }
}

/**
    cuda 其实为我们封装了一个类似于我们的 CudaAllocator 的内存分配器版本的vector
    直接使用即可
    这个universal_vector 使用了统一内存分配， 可以在 CPU 和 GPU 之间自动迁移数据

    当然也有分离的 host_vector 和 device_vector
    进行赋值的时候 host_vec = device_vec 会自动进行数据拷贝
*/

int main(){
    int n = 65535;
    thrust::universal_vector<float> X(n);
    thrust::universal_vector<float> Y(n);

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
}