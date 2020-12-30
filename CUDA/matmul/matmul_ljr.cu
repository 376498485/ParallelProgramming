#include<stdio.h>
#include<stdlib.h>

__global__ void matMulByGPU(int *cuda_a, int *cuda_b, int *cuda_c, int s1, int s2, int s4){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int row = x / s1;
    int column = x % s4;
    int value=0;
    clock_t start = clock();
    for(int i=0; i<n; i++)
        value += cuda_a[row * s2 + i] * cuda_b[i * s4 + column];
    cuda_c[row * s4 + column] = value;
    *time = clock() - start;

}
int main()
{
    int s1, s2, s3, s4;
    int THREAD_NUM = 256;
    int *cuda_a, *cuda_b, *cuda_c;

    scanf("%d %d", &s1, &s2);
    a = (int*)malloc(sizeof(int)*s1*s2);
    for(int i=0; i<s1*s2; i++)
        scanf("%d", &a[i]);
    scanf("%d %d", &s3, &s4);
    b = (int*)malloc(sizeof(int)*s3*s4);
    for(int i=0; i<s3*s4; i++)
        scanf("%d",&b[i]);
    c = (int*)malloc(sizeof(int)*s1*s4);
	
    //申请显存
    cudaMalloc((void**)&cuda_a, sizeof(int)*s1*s2);
    cudaMalloc((void**)&cuda_b, sizeof(int)*s3*s4);
    cudaMalloc((void**)&cuda_c, sizeof(int)*s1*s4);
    //将内存复制到显存
    cudaMemcpy(cuda_a, a, sizeof(int)*s1*s2, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(int)*s3*s4, cudaMemcpyHostToDevice);
    //使用GPU计算
    int blocks_num = s1*s4 / THREAD_NUM + 1;

    matMulByGPU<<<blocks_num, THREAD_NUM>>>(cuda_a, cuda_b, cuda_c, int s1, int s2, int s4);

    //将结果从显存复制回内存
    cudaMemcpy(c, cuda_c, sizeof(int)*s1*s4,cudaMemcpyDeviceToHost);
    int k = 0;
    for(int i=0; i<s1; i++){
        for(int j=0; j<s4; j++){
            if(j!=0) printf(" ");
            printf("%d",c[k++]);
            if(j==s4-1) printf("\n");
        }
    }
    //打印GPU计算的时间
    //释放内存
    free(a);
    free(b);
    free(c);
    free(c_from_gpu);
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
	
}



