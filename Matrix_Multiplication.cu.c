#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 16

void matgen(int* a, int n){
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
		a[i *n + j] = rand() % 10;
		}
	}
}

void matmult(int* a, int* b, int* c, int n){
	int i, j, k;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			int t = 0;
			for(k = 0; k < n; k++){
				t += a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = t;
		}
	}
}

int main(){
	clock_t start = clock();
	int a[N * N];
    int b[N * N];
    int c[N * N];
    matgen(a, N);
	matgen(b, N); 
    matmult(a, b, c, N);
	clock_t time_used = clock() - start;
	int i;
	for(i = 0; i < N * N; i++){
        if(i % N  == 0){
            printf("\n");
        }
        printf("%d ", c[i]);
	}
	printf("\n");
	printf("time: %ldms", time_used);
	return 0;
}
