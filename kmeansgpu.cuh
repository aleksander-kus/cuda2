#ifndef KMEANS_GPU_CUH
#define KMEANS_GPU_CUH

template <unsigned int n>
int* kmeansGpu(const float* objects, int N, int k, float** centersOutput, bool isDebug = false, float threshold = 0.001f);

#include "kmeansgpu.cu"

#endif