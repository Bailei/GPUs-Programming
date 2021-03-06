#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define DATA_SIZE 999999

int data[DATA_SIZE];

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

void GenerateNumber(int* number, int size){
	for(int i = 0; i < size; i++){
		number[i] = rand() % 10;
	}
}

__global__ void sumOfSquares(int* num, int* result, clock_t* time)
{
	int sum = 0;
	int i;
	clock_t start = clock();
	for(i = 0; i < DATA_SIZE; i++){
		sum += num[i] * num[i];
	}
	*result = sum;
	*time = clock() - start;
}


int main(){
	if(!InitCUDA()){
		return 0;
	}	

	printf("CUDA initialized.\n");

	GenerateNumber(data, DATA_SIZE);

	int* gpudata, *result;
	clock_t* time;
	cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**) &result, sizeof(int));
	cudaMalloc((void**) &time, sizeof(clock_t));
	cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

	
	sumOfSquares<<<1, 1, 0>>>(gpudata, result, time);
	int sum;
	clock_t time_used;
	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);

	printf("sum: %d time: %d\n", sum, time_used);

	return 0;
}


