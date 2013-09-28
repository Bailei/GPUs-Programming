#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define W 1600

void vector_gen(float* a, int size){
	int i;
	for(i = 0; i < size; i++){
		a[i] = rand();
	}
}

void vecAdd(float* A, float* B, float* C){
	int i;
	for(i = 0; i < W; i++){
		C[i] = A[i] + B[i];
	}
}

int main(){
	float* A, *B, *C;
	A = (float*) malloc(sizeof(int) * W);
	B = (float*) malloc(sizeof(int) * W);
	C = (float*) malloc(sizeof(int) * W);
	vector_gen(A, W);
	vector_gen(B, W);

	clock_t start, time_used;
	start = clock();
	vecAdd(A, B, C);
	time_used = clock() - start;

	int i;
	for(i = 0; i < W ; i++){
        if(i % 10  == 0){
            printf("\n");
        }
        printf("%f ", C[i]);
	}
	printf("\n");
	printf("time: %ldms\n", time_used);
	return 0;

}