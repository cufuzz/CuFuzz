## Description

Hi，developers. There is an issue where a program cannot exit properly, regarding the "cublasDgemm()" function ! I think this bug could potentially be used for a DoS attack.

## Replicate

I stumbled upon a bug while compiling a super simple `.cu` file. Here's what the `poc.cu` looks like:

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    // Variable definitions and initialization
    int m = 54, n = 167, k = 122;
    size_t size_A = m * n * sizeof(double);
    size_t size_B = n * k * sizeof(double);
    size_t size_C = m * k * sizeof(double);

    // Host memory allocation and initialization
    double *h_A = (double *)malloc(size_A);
    double *h_B = (double *)malloc(size_B);
    double *h_C = (double *)calloc(m * k, sizeof(double));

    for (int i = 0; i < m * n; i++) h_A[i] = 1.0;
    for (int i = 0; i < n * k; i++) h_B[i] = 1.0;

    // Device memory allocation
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // cuBLAS handle initialization
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication using cublasDgemm
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, \
                d_A, m, d_B, k, &beta, d_C, m);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```

The compile command is: `nvcc poc.cu -o test -lcublas`  
Run it with: `./test`

## Enverments

it is GPU RTX4090 24G/

ubuntu20.04/

cpu Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz/ 

nvidia driver 550.54, cuda version 12.4

(I only attempted it on a physical machine and did not test it in VM, such as docker)

## Impacts

* When I execute it, this program gets stuck for a long time and cannot exit normally. I had to hit `Ctrl+C` in the terminal to stop it. But strangely enough, when I tried debugging with cuda-gdb, the program ended and exited normally. 

* When I switch to other values of m, n, and k, there is no such problem.

* I suspected that my function calls might not comply with the specifications, so I consulted the official manual. As follows:

```
cublasStatus_t cublasDgemm(cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb,
int m, int n, int k,
const double *alpha,
const double *A, int lda,
const double *B, int ldb,
const double *beta,
double *C, int ldc)
```

Since I am using non transposed (i.e., CUBLAS_OP_N), it is also legal for ldc to take m.

So I think there may be an issue with CublasDgemm. Moreover, I think this bug could potentially be used for a DoS attack. 

* The compiled binary cannot be exited for a long time, as shown in the following figure:
  
  <img src="https://github.com/MPSFuzz/images/blob/master/cublasDgemm_hang_1.PNG?raw=true" title="" alt="MPSFuzz/images" data-align="inline">

* However, the binary could be exited normally with cuda-gdb
  
  ![MPSFuzz/images](https://github.com/MPSFuzz/images/blob/master/cublasDgemm_hang_3.PNG?raw=true)

## others

Thank you to your team for reviewing my submission. 

Have a nice day :) !
