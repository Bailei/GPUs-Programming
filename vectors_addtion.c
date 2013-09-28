#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define W 100
#define THREAD_NUM 512
#define BLOCK_NUM 32

bool InitCUDA(){
	int count;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&count);
	if(count == 0){
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++){
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){
			if(prop.major >= 1){
				break;
			}
		}
	} 

	if(i == count){
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

void vector_gen(int* a, int size){
	for(int i = 0; i < size; i++){
		a[i] = rand() % 10;
	}
}

void vecAdd(float* A, float* B, float* C, int n){
	int size = n * sizeof(float);
	float* A_d, *B_d, *C_d;

	cudaMalloc((void**) &A_d, size);
	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &B_d, size);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &C_d, size);

	vecAddkernel<<<1, 100>>>(A_d, B_d, C_d, W);
__global__ void vecAddkernel(float* A_d, float* B_d, float* C_d, int n){
	int tx = threadIdx.x;
	int i = tx;
	if(i < n) C_d[i] = A_d[i] + B_d[i];
}
	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}


int main(){
	if(!InitCUDA()){
		return 0;
	}

	float* A, *B, *C;
	A = (float*) malloc(sizeof(int) * W);
	B = (float*) malloc(sizeof(int) * W);
	C = (float*) malloc(sizeof(int) * W);
	vector_gen(A);
	vector_gen(B);
	
	vecAdd(A, B, C, W);
	for(int i = 0; i < X * X; i++){
        if(i % X  == 0){
            printf("\n");
        }
        printf("%d ", C[i]);
	}
	printf("\n");
	// printf("time: %ldms", time_used);
	printf("CUDA initialized.\n");


	return 0;
}