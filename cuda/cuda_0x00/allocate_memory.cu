#include<cstdio>
#include<cuda_runtime.h>

__device__ __host__ void cuthead(char* p){
    (*p)++;
}

__global__ void kernel(char* p){
    printf("Hello from GPU\n");
    cuthead(p);
}

int main(){
    /**
    char p = 1;
    cuthead(&p);

    char* g_p;
    cudaMalloc(&g_p, sizeof(char));
    cudaMemcpy(g_p, &p, sizeof(char), cudaMemcpyHostToDevice);
    kernel<<<1,1>>>(g_p);    
    char res;
    //cudaMemcpy会自动进行同步
    cudaMemcpy(&res, g_p, sizeof(char), cudaMemcpyDeviceToHost);
    printf("res : %d\n", res);
    cudaFree(&g_p);
    */

    
    char* g_p;
    // 新显卡上支持的特性， 自动的进行CPU和GPU之间的内存拷贝
    // 并非完全没有开销， 尽量使用分离的内存管理
    cudaMallocManaged(&g_p, sizeof(char));
    kernel<<<1,1>>>(g_p);
    cudaDeviceSynchronize();
    printf("res : %d\n", *g_p); 
    cudaFree(&g_p);
}