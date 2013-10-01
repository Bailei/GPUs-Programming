#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

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
    
    int i;
    for(i = 100; i < 200000000; i *= 2){
    clock_t start = clock();
    float* A, *B, *C;
	A = (float*) malloc(sizeof(float) * i);
	B = (float*) malloc(sizeof(float) * i);
	C = (float*) malloc(sizeof(float) * i);
	vector_gen(A, i);
	vector_gen(B, i);
    int size = i * sizeof(float);
	float* A_d, *B_d, *C_d;

	cudaMalloc((void**) &A_d, size);
	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &B_d, size);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &C_d, size);
    
	dim3 dimBlock(512, 1);
	dim3 dimGrid(ceil(float(i)/512), 1);
	vecAddkernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, i);

	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	clock_t end = (clock() - start) / 1000;
	printf("Length %d vector addion use time: %ldms\n", i, end);
    }
	return 0;
}
