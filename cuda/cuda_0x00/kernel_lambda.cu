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

template<typename Func>
__global__ void parallel_for(int n, Func f){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        f(i);
    }
}


int main(){
    int n = 10;
    std::vector<int, CudaAllocator<int>> data(n);

    // 这里的data是cpu上的地址， 直接捕获的话并不能捕获到GPU上的地址
    /*
    parallel_for<<<1, 10>>>(n, [&] __device__ (int i){
        data[i] = i + 1;
    });
    */

    //这里会直接发生深拷贝， 导致整个vec被复制到GPU上
    /*
    parallel_for<<<2, 5>>>(n, [=] __device__ (int i){
        data[i] = i * i;
    });
    */

    // 正确的做法是捕获data的指针
    int * ptr = data.data();
    parallel_for<<<2, 5>>>(n, [=] __device__ (int i){
        ptr[i] = i;
    });
    cudaDeviceSynchronize();

    for(int i = 0; i < n; i++){
        printf("data[%d] = %d\n", i, data[i]);
    }
    
}