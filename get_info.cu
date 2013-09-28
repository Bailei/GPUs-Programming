#include <stdio.h>
#include <cuda_runtime.h>

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

	printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
	printf("clockRate: %ld\n", prop.clockRate);
	printf("totalGlobalMem: %ld\n", prop.totalGlobalMem);

	return true;
}

int main(){
	InitCUDA();
	printf("CUDA initialized.\n");

	return 0;
}