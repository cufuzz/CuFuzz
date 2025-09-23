#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "/home/fanximing/cuda-graph-llm/c_factors/mutate.h"


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <nvjpeg.h>

void UT(const unsigned char* jpegData, size_t jpegSize, nvjpegHandle_t nvjpegHandle, nvjpegJpegStream_t jpegStream, cudaStream_t stream) {
    if (jpegData == NULL || jpegSize == 0) return;

    unsigned int width, height;
    nvjpegChromaSubsampling_t subsampling;

    nvjpegStatus_t status = nvjpegJpegStreamGetFrameDimensions(jpegStream, &width, &height);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return;
    }

    status = nvjpegJpegStreamGetChromaSubsampling(jpegStream, &subsampling);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return;
    }

    // Additional processing can be added here if needed
}

int main(int argc, char* argv[]) {
    int loops = 0;
    while (loops < 2000) {
    const char* filename;
    if (argc > 1) {
        filename = argv[1];
    } else {
        return EXIT_FAILURE;
    }

    nvjpegHandle_t nvjpegHandle;
    nvjpegJpegStream_t jpegStream;
    cudaStream_t stream;

    // Initialize resources
    if (nvjpegCreateSimple(&nvjpegHandle) != NVJPEG_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (nvjpegJpegStreamCreate(nvjpegHandle, &jpegStream) != NVJPEG_STATUS_SUCCESS) {
        nvjpegDestroy(nvjpegHandle);
        return EXIT_FAILURE;
    }
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        nvjpegJpegStreamDestroy(jpegStream);
        nvjpegDestroy(nvjpegHandle);
        return EXIT_FAILURE;
    }

    // Open and read file
    FILE* file = fopen(filename, "rb");
    if (!file) {
        nvjpegJpegStreamDestroy(jpegStream);
        nvjpegDestroy(nvjpegHandle);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    //initialization
    unsigned char* jpgData = (unsigned char*)malloc(fileSize);
    size_t bytesRead = fread(jpgData, 1, fileSize, file);
    fclose(file);
    //initialization

    // wrap buffer

    // wrap havoc and writing .cur
    char =  (unsigned char*)malloc(fileSize);

    bytesRead =  fread(jpgData, 1, fileSize, file);

    u32 jpgData_len = fileSize;
    u8 *jpgData_buf = (u8 *)malloc(jpgData_len);
    u32 jpgData_len_havoc;
    jpgData_len_havoc = random_havoc(jpgData_buf, jpgData_len, 1);
    char *temp_1 = (char *)realloc(jpgData, fileSize);
    jpgData = temp_1;
    memcpy((u8*)jpgData, jpgData_buf, jpgData_len);
    FILE *file_1 = fopen("1.bin", "wb");
    fwrite(jpgData_buf, jpgData_len, 1, file_1);
    fclose(file_1);

    // wrap all done!

    if (bytesRead != fileSize) {
        free(jpgData);
        nvjpegJpegStreamDestroy(jpegStream);
        nvjpegDestroy(nvjpegHandle);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    // Process the image
    UT(jpgData, fileSize, nvjpegHandle, jpegStream, stream);

    // Cleanup
    free(jpgData);
    cudaStreamDestroy(stream);
    nvjpegJpegStreamDestroy(jpegStream);
    nvjpegDestroy(nvjpegHandle);

    free(jpgData_buf);
    loops++;}
    return EXIT_SUCCESS;
}
