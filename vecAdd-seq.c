#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* A, *B, *C;

void vector_gen(float* a, int size){
	int i;
	for(i = 0; i < size; i++){
		a[i] = rand();
	}
}

void vecAdd(float* A, float* B, float* C, int x){
	int i;
	for(i = 0; i < x; i++){
		C[i] = A[i] + B[i];
	}
}

int main(){
	int i;
    for(i = 100; i < 200000000; i *= 2){
    A = (float*) malloc(sizeof(float) * i);
	B = (float*) malloc(sizeof(float) * i);
	C = (float*) malloc(sizeof(float) * i);
    vector_gen(A, i);
	vector_gen(B, i);
    clock_t start = clock();
	vecAdd(A, B, C, i);
	clock_t end = (clock() - start) / 1000;
	printf("Length %d vector addion use time: %ldms\n", i, end);
    }
	return 0;

}
