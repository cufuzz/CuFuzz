#include <fcntl.h>
#include <cmath>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "/home/fanximing/cuda-graph-llm/c_factors/mutate.h"


#include <stdio.h>
#include <stdlib.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

// CUDA API sequence to be tested
void UT(cusolverDnHandle_t handle, int m, int n, double *d_A, int lda) {
    int Lwork_geqrf;

    // 1. Query buffer size for cusolverDnDgeqrf
    cusolverDnDgeqrf_bufferSize(handle, m, n, d_A, lda, &Lwork_geqrf);

    // Device synchronization
    cudaDeviceSynchronize();
}

int main() {
    int loops = 0;
    while (loops < 2000) {
        printf("%d\n",loops);
    //initialization
    int m = 5;
    int n = 5;
    int lda = m;
    size_t size = m * n * sizeof(double);
    double *h_A = (double *)malloc(size);
    for (int i = 0; i < m * n; i++) {
        h_A[i] = (double)(i + 1);
    }
    //initialization

    // wrap buffer
    u32 m_len = sizeof(m);
    u8 *m_buf = (u8 *)malloc(m_len);
    memcpy(m_buf, &m, m_len);
    u32 n_len = sizeof(n);
    u8 *n_buf = (u8 *)malloc(n_len);
    memcpy(n_buf, &n, n_len);

    // wrap havoc and writing .cur
    u32 m_len_havoc;
    u32 n_len_havoc;
    m_len_havoc = random_havoc(m_buf, m_len, 0);
    m = *(int *)m_buf;
    m %= 4096;
    if (m<0) { m = -m ;}
    if (m==0) { m += 1 ;}
    FILE *file_1 = fopen("1.bin", "wb");
    fwrite(m_buf, m_len, 1, file_1);
    fclose(file_1);
    n_len_havoc = random_havoc(n_buf, n_len, 0);
    n = *(int *)n_buf;
    n %= 4096;
    if (n<0) { n = -n ;}
    if (n==0) { n += 1 ;}
    FILE *file_2 = fopen("2.bin", "wb");
    fwrite(n_buf, n_len, 1, file_2);
    fclose(file_2);
    lda =  m;

    size =  m * n * sizeof(double);

    u32 h_A_len = size;
    u8 *h_A_buf = (u8 *)malloc(h_A_len);
    u32 h_A_len_havoc;
    h_A_len_havoc = random_havoc(h_A_buf, h_A_len, 1);
    double *temp_3 = (double *)realloc(h_A, size);
    h_A = temp_3;
    memcpy((u8*)h_A, h_A_buf, h_A_len);
    FILE *file_3 = fopen("3.bin", "wb");
    fwrite(h_A_buf, h_A_len, 1, file_3);
    fclose(file_3);

    // wrap all done!
    
    // Device memory allocation
    double *d_A = nullptr;
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Create cusolver handle
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Execute the CUDA API sequence
    UT(handle, m, n, d_A, lda);

    // Cleanup
    free(h_A);
    cudaFree(d_A);
    cusolverDnDestroy(handle);

    free(m_buf);
    free(n_buf);
    free(h_A_buf);
    loops++;}
    return EXIT_SUCCESS;
}
