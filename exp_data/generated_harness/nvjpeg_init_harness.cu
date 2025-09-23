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

    // Read file data
    unsigned char* jpgData = (unsigned char*)malloc(fileSize);
    size_t bytesRead = fread(jpgData, 1, fileSize, file);
    fclose(file);

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

    return EXIT_SUCCESS;
}
