#ifndef KMEANS_GPU_CUH
#define KMEANS_GPU_CUH

// #define BOARDSIZE 9
// #define BOARDLENGTH 81
// #define BLANK 0

// __host__ __device__ bool findEmpty(const char* board, int& i, int& j);

// __host__ __device__ bool tryToInsertBox(const char* board, const int& i, const int& j, const char& value);

// __host__ __device__ bool tryToInsert(const char* board, const int& i, const int& j, const char& value);

template <unsigned int n>
int* kmeansGpu(const float* objects, int N, int k, float** centersOutput, float threshold = 0.001f);

#include "kmeansgpu.cu"

#endif