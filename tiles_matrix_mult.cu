/*
    Matrix Multiplication on GPU with tiles
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>

#include "common.h"

#define MATRIX_SIZE 2000 // Matrix size
#define TILE_SIZE 8 // tiles size

// Fill the matrix
void fillMatrix(float * M){
    int size = MATRIX_SIZE * MATRIX_SIZE;
    for(int i = 0; i < size; i++){
        M[i] = (float)rand()/(RAND_MAX/ 10.0f);
    }
    return;
}

// Print the matrix
void printMatrix(float * M){
    int size = MATRIX_SIZE * MATRIX_SIZE;
    for(int x = 0; x < size; x++){
        if(x % MATRIX_SIZE == 0){
            printf("\n");
        }
        printf("%f ", M[x]);
    }
}

// Check results
int checkResult(float * hostRef, float * gpuRef){
    double epsilon = 0.5;
    bool match = 1;
    int size = MATRIX_SIZE * MATRIX_SIZE;
    for (int i = 0; i < size; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("host %f gpu %f dif %f\n", hostRef[i], gpuRef[i],hostRef[i] - gpuRef[i]);
            break;
        }
    }
    return match;
}

// Multiply in CPU
void multiplyMatrixCPU(float * C, float * A, float * B){
    for(int y = 0; y < MATRIX_SIZE; y++){
        for(int z = 0; z < MATRIX_SIZE; z++){
            for(int x = 0; x < MATRIX_SIZE; x++){
                C[y * MATRIX_SIZE + z] += A[x + y * MATRIX_SIZE] * B[x * MATRIX_SIZE + z];
            }
        }
    }
}

// Multiply in GPU no tiles
__global__ void multiplyMatrixGPU(float * A, float * B, float * C){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    float sum = 0;
    if (ix < MATRIX_SIZE && iy < MATRIX_SIZE){
        for(int j = 0; j < MATRIX_SIZE; j++){
            sum += A[ix * MATRIX_SIZE + j] * B[j * MATRIX_SIZE + iy];
        }
        C[ix * MATRIX_SIZE + iy] = sum;
    }
}

// Multiplicar GPU con tiles
__global__ void multiplyMatrixGPUTiles(float * A, float * B, float * C){
    float sum = 0;
    unsigned int ix = threadIdx.x + TILE_SIZE * blockIdx.x;
    unsigned int iy = threadIdx.y + TILE_SIZE * blockIdx.y;
    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;
    // Shared variables
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    sharedA[y][x] = 0.0; // inicia 0
    sharedB[y][x] = 0.0; // inicia 0

    __syncthreads();

    // i-- para seguir con los 0
    for (int i = (TILE_SIZE + MATRIX_SIZE - 1) / TILE_SIZE; i >= 0; i--) {
        if ((i * TILE_SIZE + x ) < MATRIX_SIZE && iy < MATRIX_SIZE) {
            sharedA[y][x] = A[iy * MATRIX_SIZE + i * TILE_SIZE + x];
        }
        if ((i * TILE_SIZE + y) < MATRIX_SIZE && ix < MATRIX_SIZE){
            sharedB[y][x] = B[(i * TILE_SIZE + y) * MATRIX_SIZE + ix];
        }

        __syncthreads(); // sync

        for (int j = 0; j < TILE_SIZE; j++){
             sum += sharedA[y][j] * sharedB[j][x];
        }
        __syncthreads(); // sync
    }

    if (ix < MATRIX_SIZE && iy < MATRIX_SIZE){
      C[iy * MATRIX_SIZE + ix] = sum;
    }
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = MATRIX_SIZE;
    int ny = MATRIX_SIZE;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: x %d y %d\n", nx, ny);
    std::cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE <<'\n';
    std::cout << '\n';


    // malloc host memory
    float *h_m1, *h_m2, *hostRef, *gpuRef, *gpuRefTiles;
    h_m1 = (float *)malloc(nBytes);
    h_m2 = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    gpuRefTiles = (float *)malloc(nBytes);

    fillMatrix(h_m1); // initialize data at host side
    fillMatrix(h_m2); // initialize data at host side

    memset(hostRef, 0, nBytes);
    memset(gpuRefTiles, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    ////////// CPU for reference //////////
    auto start_cpu =  std::chrono::high_resolution_clock::now();
    multiplyMatrixCPU(hostRef, h_m1, h_m2);
    auto end_cpu =  std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multiplyMatrixCPU elapsed %f ms\n\n", duration_ms.count());

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_m1, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_m2, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = TILE_SIZE;
    int dimy = TILE_SIZE;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    ////////// GPU no tiles //////////
    start_cpu =  std::chrono::high_resolution_clock::now();
    multiplyMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  std::chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;

    printf("multiplyMatrixGPU <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    if(checkResult(hostRef, gpuRef))
      printf("MATCH\n\n");
    else
      printf("NO MATCH\n\n");

    ////////// GPU with tiles //////////
    start_cpu =  std::chrono::high_resolution_clock::now();
    multiplyMatrixGPUTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatC);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  std::chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;

    printf("multiplyMatrixGPUTiles <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,grid.y,block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRefTiles, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    if(checkResult(hostRef, gpuRefTiles))
      printf("MATCH\n\n");
    else
      printf("NO MATCH\n\n");

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_m1);
    free(h_m2);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
