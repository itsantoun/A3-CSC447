#include <stdio.h>
#include <stdlib.h>

#define M 4
#define N 3
#define K 4

void matrixMul(int *a, int *b, int *c) {
    #pragma acc parallel loop collapse(2) present(a, b, c)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main() {
    int *h_a, *h_b, *h_c;  // host matrices
    int size = M * N * sizeof(int);

    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    for (int i = 0; i < M * N; ++i) {
        h_a[i] = 1; 
        h_b[i] = 2; 
    }

    // Call matrix multiplication function with OpenACC
    matrixMul(h_a, h_b, h_c);


    printf("Result Matrix (Basic):\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_c[i * N + j]);
        }
        printf("\n");
    }

    free(h_a);
 free(h_b); 
free(h_c);

    return 0;
}

