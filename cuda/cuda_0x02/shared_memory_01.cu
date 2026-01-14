#include"../cuda_allocator.cuh"

__global__ void kernel(int* sum, int const* data, int n){
    __shared__ __volatile__ int local_sum[1024];
    int i = blockIdx.x;
    int j = threadIdx.x;
    int temp_sum = 0;
    for(int t = i * 1024 + j; t < n; t += gridDim.x * 1024){
        temp_sum += data[t];
    }

    local_sum[j] = temp_sum;
    __syncthreads();
    // 规约操作
    if (j < 512){
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256){
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128){
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64){
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    /** version_1这样的形式会有线程组分歧。线程组分歧..去看笔记
    if (j < 32){
        local_sum[j] += local_sum[j + 32];
    }
    
    if (j < 16){
        local_sum[j] += local_sum[j + 16];
    }
    
    if (j < 8){
        local_sum[j] += local_sum[j + 8];
    }
    
    if (j < 4){
        local_sum[j] += local_sum[j + 4];
    }
    
    if (j < 2){
        local_sum[j] += local_sum[j + 2];
    }

    if (j == 0){
        sum[i] = local_sum[0] + local_sum[1];
    }
    */
    if (j < 32){
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];

        if (j == 0){
            sum[i] += local_sum[0] + local_sum[1];
        }
    }

    /**
        不过这里计算的结果似乎还是不对,这是因为Shared Memory执行一个板块内的线程时，并不是全部同时执行
        ，而是一会执行一个线程，一会执行另一个。所以有的线程会执行到 j < 32这条判断，而另一个线程的规约
        操作还未完成。(为什么这样设计？ 因为线程可能在等待数据抵达，如果一直等待则会导致性能的损失CPU中
        这部分叫做超线程技术)。
        
        那么如何解决这个问题呢？
        加上__syncthreads()

        SM的整体调度是由32个线程为一组进行调度的， 因此编号小于32的线程不需要进行__syncthreads()
    */
}

int main(){
    int n = 65536 * 2;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> result(n / 1024);

    for(int i = 0; i < n;++i){
        arr[i] = 1;
    }
    kernel<<<n / 1024, 1024>>>(result.data(), arr.data(), n);
    cudaDeviceSynchronize();
    int final_sum = 0;

    for(int i = 0; i < result.size();++i){
        final_sum += result[i];
    }
    printf("Final sum: %d\n", final_sum);
}