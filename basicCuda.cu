#include <stdio.h>
#include <stdlib.h>

// Define matrix dimensions
#define M 4
#define N 3
#define K 4

// CUDA kernel for matrix multiplication without tiling
__global__ void matrixMul(int *a, int *b, int *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    int *h_a, *h_b, *h_c;  
    int *d_a, *d_b, *d_c; 
    int size = M * N * sizeof(int);

    // Allocate host memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) {
        h_a[i] = 1; // Initialize with 1 for simplicity
        h_b[i] = 2; // Initialize with 2 for simplicity
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy host data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

   
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);


    printf("Result Matrix:\n");
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

