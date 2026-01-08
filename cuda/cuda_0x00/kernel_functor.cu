#include<cstdio>
#include<cuda_runtime.h>

/**
    核函数可以接受一个仿函数作为参数
    通过模板参数传递仿函数类型
*/

struct Functor{
    __device__ void operator()(int i){
        printf("Hello from functor, param %d \n", i);
    }
};

template<typename Func>
__global__ void kernel(int n, Func f){
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x){
        f(i);
    }
}

int main(){
    int n = 20;
    kernel<<<1, 4>>>(n, Functor{});
    cudaDeviceSynchronize();

    // 既然传入仿函数， 那么也可以传入一个lambda表达式
    // 注意需要编译时加入--extended-lambda参数
    kernel<<<1, 4>>>(n, [] __device__ (int i){
        printf("Hello from lambda, param %d \n", i);
    });


    cudaDeviceSynchronize();
    return 0;
}
