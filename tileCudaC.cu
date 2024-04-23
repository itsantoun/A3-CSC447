#include <stdio.h>
#include <stdlib.h>

// Define matrix dimensions
#define M 4
#define N 3
#define K 4
#define TILE_WIDTH 10

// CUDA kernel for matrix multiplication and tiling
__global__ void matrixMulTiling(int *a, int *b, int *c, int width) {
    __shared__ int ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_b[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int sum = 0;
    for (int t = 0; t < width / TILE_WIDTH; ++t) {
        ds_a[threadIdx.y][threadIdx.x] = a[row * width + t * TILE_WIDTH + threadIdx.x];
        ds_b[threadIdx.y][threadIdx.x] = b[(t * TILE_WIDTH + threadIdx.y) * width + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += ds_a[threadIdx.y][k] * ds_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < width && col < width) {
        c[row * width + col] = sum;
    }
}

int main() {
    int *h_a, *h_b, *h_c;  
    int *d_a, *d_b, *d_c;  
    int size = M * N * sizeof(int);
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) {
        h_a[i] = 1; 
        h_b[i] = 2; 
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    matrixMulTiling<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

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
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    return 0;
}

