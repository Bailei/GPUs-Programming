#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define X 16

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

void matgen(float* a, int n){
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			a[i * n + j] = (float)rand();
		}
	}
}	

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, clock_t* time){
	
	int tx = threadIdx.x;
	int ty = threadIdy.y;
	float Pvalue = 0;
	clock_t start = clock();
	for(int k = 0; k < X; k++){
		float Mdelement = Md[ty * X + k];
		float Ndelement = Nd[k * X + tx];
		Pvalue += Mdelement * Ndelement;
	}
	Pd[ty * X + tx] = Pvalue;
	*time = clock() - start;
}


int main(){
	if(!InitCUDA())
		return 0;
	printf("CUDA initialized.\n");

	clock_t* time;
	float* M, *N, *P;
    M = (float*) malloc(sizeof(float) * X * X);
    N = (float*) malloc(sizeof(float) * X * X);
    P = (float*) malloc(sizeof(float) * X * X);
	
	srand(0);
	matgen(M, X);
	matgen(N, X);

	int size = X * X * sizeof(float);
	float* Md, *Nd, *Pd;

	cudaMalloc((void**) &Md, size);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &Nd, size);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &Pd, size);
	cudaMalloc((void**) &time, sizeof(clock_t));

	dim3 dimBlock(X, X);
	dim3 dimGrid(1, 1);
	MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, time);

	clock_t time_used;
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);

	for(int i = 0; i < X * X; i++){
        if(i % X  == 0){
            printf("\n");
        }
        printf("%d ", P[i]);
	}
	printf("\n");
	printf("time: %ldms", time_used);

	return 0;
}