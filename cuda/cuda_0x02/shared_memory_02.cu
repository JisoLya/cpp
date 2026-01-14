#include"../cuda_allocator.cuh"

/**
    经过我们01部分的解释，我们可以直接对这个数组求和的代码进行模板包装
*/

template<int blockSize, typename T>
__global__ void parallel_sum(T *sum, T const* data, int n){
    __shared__ volatile T local_sum[blockSize];
    

    int i = threadIdx.x;
    int j = blockIdx.x;
    //load data
    int temp_sum = 0;
    for(int t = i + j * blockSize; t < n;t += gridDim.x * blockSize){
        temp_sum += data[t];
    }
    local_sum[i] = temp_sum;
    __syncthreads();

    if constexpr (blockSize >= 1024){
        if (i < 512){
            local_sum[i] += local_sum[i + 512];
        }
        __syncthreads();
    }

    if constexpr (blockSize >= 512){
        if (i < 256){
            local_sum[i] += local_sum[i + 256];
        }
        __syncthreads();
    }

    if constexpr (blockSize >= 256){
        if (i < 128){
            local_sum[i] += local_sum[i + 128];
        }
        __syncthreads();
    }

    if constexpr (blockSize >= 128){
        if (i < 64){
            local_sum[i] += local_sum[i + 64];
        }
        __syncthreads();
    }

    if (i < 32){
        if constexpr (blockSize >= 64)
            local_sum[i] += local_sum[i + 32];
        if constexpr (blockSize >= 32)
            local_sum[i] += local_sum[i + 16];
        if constexpr (blockSize >= 16)
            local_sum[i] += local_sum[i + 8];
        if constexpr (blockSize >= 8)
            local_sum[i] += local_sum[i + 4];
        if constexpr (blockSize >= 4)
            local_sum[i] += local_sum[i + 2];

        if (i == 0){
            sum[j] = local_sum[0] + local_sum[1];
        }
    }
}

template<int reduceScale = 4096, int blockSize = 256, typename T>
T array_sum(T const* data, int n){
    
    std::vector<T, CudaAllocator<T>> block_sums(n / reduceScale);

    parallel_sum<blockSize><<<n / reduceScale, blockSize>>>(block_sums.data(), data, n);
    cudaDeviceSynchronize();
    T host_sum = 0;
    for(int i = 0;i < block_sums.size();i++){
        host_sum += block_sums[i];
    }
    return host_sum;
}


int main(){
    int n = 65536 * 2;
    std::vector<int, CudaAllocator<int>> data(n);
    for(int i = 0;i < n;i++){
        data[i] = 1;
    }
    
    int sum = array_sum(data.data(), n);
    printf("Sum: %d\n", sum);
    return 0;
}