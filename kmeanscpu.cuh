#ifndef KMEANS_CPU_CUH
#define KMEANS_CPU_CUH

// #define BOARDSIZE 9
// #define BOARDLENGTH 81
// #define BLANK 0

// __host__ __device__ bool findEmpty(const char* board, int& i, int& j);

// __host__ __device__ bool tryToInsertBox(const char* board, const int& i, const int& j, const char& value);

// __host__ __device__ bool tryToInsert(const char* board, const int& i, const int& j, const char& value);

template <unsigned int n>
int* kmeansCpu(const float* objects, int N, int k);

#include "kmeanscpu.cu"

#endif