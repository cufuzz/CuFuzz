#include <fcntl.h>
#include <cmath>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>               // **[新增] 用于 wall-clock 计时**
#include "/home/fanximing/cuda-graph-llm/c_factors/mutate.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA API sequence to be tested
void UT(cudaStream_t *stream, int *data, int numElements) {
    cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
    cudaStreamBeginCapture(*stream, cudaStreamCaptureModeGlobal);

    cudaStreamCaptureStatus captureStatus;
    unsigned long long id;
    cudaStreamGetCaptureInfo(*stream, &captureStatus, &id);

    cudaStreamEndCapture(*stream, nullptr);
}

int main() {
    // ================== 开始计时 ==================
    struct timespec start, end;                     // **[新增]**
    clock_gettime(CLOCK_MONOTONIC, &start);         // **[新增]**

    int loops = 0;
    while (true) {
        printf("%d\n", loops);                      // **[新增]**

        // initialization
        int numElements = 256;
        size_t size = numElements * sizeof(int);
        int *h_data = (int *)malloc(size);
        for (int i = 0; i < numElements; i++) {
            h_data[i] = i;
        }

        // wrap buffer
        u32 numElements_len = sizeof(numElements);
        u8 *numElements_buf = (u8 *)malloc(numElements_len);
        memcpy(numElements_buf, &numElements, numElements_len);

        // havoc
        u32 numElements_len_havoc;
        numElements_len_havoc = random_havoc(numElements_buf, numElements_len, 0);
        numElements = *(int *)numElements_buf;
        numElements %= 4096;
        if (numElements < 0) numElements = -numElements;
        if (numElements == 0) numElements += 1;

        
        FILE *file_1 = fopen("1.bin", "wb");
        fwrite(numElements_buf, numElements_len, 1, file_1);
        fclose(file_1);
        size = numElements * sizeof(int);

        // u32 h_data_len = size;
        u32 h_data_len = 4096* sizeof(int);
        u8 *h_data_buf = (u8 *)malloc(h_data_len);
        u32 h_data_len_havoc;
        h_data_len_havoc = random_havoc(h_data_buf, h_data_len, 1);
        int *temp_2 = (int *)realloc(h_data, size);
        h_data = temp_2;
        memcpy((u8 *)h_data, h_data_buf, size);
        FILE *file_2 = fopen("2.bin", "wb");
        fwrite(h_data_buf, h_data_len, 1, file_2);
        fclose(file_2);


        // ---------------- 触发条件 ----------------
        // if ( h_data[0] >0 && h_data[0] < 10) {   // **[新增]**
        //     clock_gettime(CLOCK_MONOTONIC, &end);           // **[新增]**
        //     double elapsed = (end.tv_sec  - start.tv_sec) +
        //                      (end.tv_nsec - start.tv_nsec) / 1e9;
        //     printf("Trigger condition met! numElements = %d\n", h_data[0]);
        //     printf("Wall-clock time since start: %.6f seconds\n", elapsed);
        //     exit(EXIT_SUCCESS);                             // **[新增]**
        // }
        if ( h_data[0]> 900 && h_data[0] < 1000) {   // **[新增]**
            clock_gettime(CLOCK_MONOTONIC, &end);           // **[新增]**
            double elapsed = (end.tv_sec  - start.tv_sec) +
                             (end.tv_nsec - start.tv_nsec) / 1e9;
            printf("Trigger condition met! numElements = %d\n", h_data[0]);
            printf("Wall-clock time since start: %.6f seconds\n", elapsed);
            exit(EXIT_SUCCESS);                             // **[新增]**
        }
        // -----------------------------------------


        int *d_data = nullptr;
        cudaMalloc((void **)&d_data, size);

        cudaStream_t stream;
        UT(&stream, h_data, numElements);

        cudaStreamDestroy(stream);
        free(h_data);
        cudaFree(d_data);
        free(numElements_buf);
        free(h_data_buf);
        loops++;
    }
    return EXIT_SUCCESS;
}
