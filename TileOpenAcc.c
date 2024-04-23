#include <stdio.h>
#include <stdlib.h>

#define M 4
#define N 3
#define K 4
#define TILE_WIDTH 10

void matrixMulTiling(int *a, int *b, int *c) {
    #pragma acc parallel present(a, b, c)
    {
        int ds_a[TILE_WIDTH][TILE_WIDTH];
        int ds_b[TILE_WIDTH][TILE_WIDTH];

        #pragma acc loop collapse(2)
        for (int i = 0; i < M / TILE_WIDTH; ++i) {
            for (int j = 0; j < N / TILE_WIDTH; ++j) {
                int row = i * TILE_WIDTH + threadIdx.y;
                int col = j * TILE_WIDTH + threadIdx.x;

                int sum = 0;
                for (int t = 0; t < K / TILE_WIDTH; ++t) {
                    ds_a[threadIdx.y][threadIdx.x] = a[row * K + t * TILE_WIDTH + threadIdx.x];
                    ds_b[threadIdx.y][threadIdx.x] = b[(t * TILE_WIDTH + threadIdx.y) * N + col];
                    #pragma acc barrier
                    #pragma acc loop
                    for (int k = 0; k < TILE_WIDTH; ++k) {
                        sum += ds_a[threadIdx.y][k] * ds_b[k][threadIdx.x];
                    }
                    #pragma acc barrier
                }

                if (row < M && col < N) {
                    c[row * N + col] = sum;
                }
            }
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

    matrixMulTiling(h_a, h_b, h_c);

    printf("Result Matrix (Tiled):\n");
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

