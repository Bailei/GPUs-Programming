#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define W 5000000

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

__global__ void vecAddkernel(float* A_d, float* B_d, float* C_d, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	C_d[i] = A_d[i] + B_d[i];
}

int main(){
	if(!InitCUDA()){
		return 0;
	}
	printf("CUDA initialized.\n");

	clock_t start = clock();
	float* A, *B, *C;
	A = (float*) malloc(sizeof(int) * W);
	B = (float*) malloc(sizeof(int) * W);
	C = (float*) malloc(sizeof(int) * W);
	vector_gen(A, W);
	vector_gen(B, W);
	
	int size = W * sizeof(float);
	float* A_d, *B_d, *C_d;

	cudaMalloc((void**) &A_d, size);
	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &B_d, size);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &C_d, size);

	dim3 dimBlock(512, 1);
	dim3 dimGrid(ceil(float(W)/512), 1);

	vecAddkernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, W);
	clock_t end = clock() - start;

	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	for(int i = 0; i < W ; i++){
        if(i % 400  == 0){
            //printf("\n");
            //printf("\n");
        }
        //printf("%f ", C[i]);
	}
	printf("\n");
	printf("time: %ldms\n", end);
	return 0;
}