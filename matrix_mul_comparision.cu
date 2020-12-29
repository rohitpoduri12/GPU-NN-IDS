#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define BLOCK_SIZE 64
#define WA 64 * BLOCK_SIZE
#define HA 16 * BLOCK_SIZE
#define WB 16 * BLOCK_SIZE
#define HB WA
#define WC WB
#define HC HA

__global__ void
matrixMul_coalescing( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[tx][ty] = B[b + wB * ty + tx];


	__syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
          Csub += As[ty][k] * Bs[tx][k];

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


__global__ void
matrixMul_tiling( float* C, float* A, float* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];


    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

	As[ty][tx] = A[a + wA * ty + tx];
        Bs[tx][ty] = B[b + wB * tx + ty];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    	C[c + wB * ty + tx] = Csub;
}



__global__ void
matrixMul_naive( float* C, float* A, float* B, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = by * blockDim.y + ty;
  int j = bx * blockDim.x + tx;

  float accu = 0.0;

  for(int k=0; k<wA; k++){
    accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
  }

  C[ i * wB + j ] = accu;

}


 void Init(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = sin(i);
}

void display(float* matrix, int size)
{

    for(int i = 0; i < size; i++)
	printf("\n%f",matrix[i]);
}


int main()
{

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // initialize host memory
    Init(h_A, size_A);
    Init(h_B, size_B);
    
    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

    dim3 threads,grid;
    
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    //matrixMul_naive<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    //matrixMul_coalescing<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    //matrixMul_tiling<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}




