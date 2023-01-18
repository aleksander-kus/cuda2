#ifndef KMEANS_CPU_CUH
#define KMEANS_CPU_CUH

template <unsigned int n>
int* kmeansCpu(const float* objects, int N, int k, float** centersOutput, float threshold = 0.001f);

#include "kmeanscpu.cu"

#endif