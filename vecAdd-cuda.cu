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

void vector_gen(float* a, int size){
	for(int i = 0; i < size; i++){
		a[i] = rand();
	}
}

__global__ void vecAddkernel(float* A_d, float* B_d, float* C_d, int n, clock_t* time){
	clock_t start = clock();
	int tx = threadIdx.x;
	int i = tx;
	if(i < n) C_d[i] = A_d[i] + B_d[i];
	*time = clock() - start;
}

int main(){
	if(!InitCUDA()){
		return 0;
	}
	printf("CUDA initialized.\n");

	float* A, *B, *C;
	A = (float*) malloc(sizeof(int) * W);
	B = (float*) malloc(sizeof(int) * W);
	C = (float*) malloc(sizeof(int) * W);
	vector_gen(A, W);
	vector_gen(B, W);
	
	int size = W * sizeof(float);
	float* A_d, *B_d, *C_d;
	clock_t *time;
	clock_t time_used;

	cudaMalloc((void**) &A_d, size);
	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &B_d, size);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &time, sizeof(clock_t));

	cudaMalloc((void**) &C_d, size);

	vecAddkernel<<<1, W>>>(A_d, B_d, C_d, W, time);

	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	for(int i = 0; i < W ; i++){
        if(i % 10  == 0){
            printf("\n");
        }
        printf("%f ", C[i]);
	}
	printf("\n");
	printf("time: %ldms\n", time_used);
	return 0;
}