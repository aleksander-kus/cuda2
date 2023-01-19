#include "cuda_runtime.h"

#include <iostream>
#include <chrono>

#include "kmeanscpu.cuh"

#define BLOCK_SIZE 1024

#define ERR(status) { \
    if (status != cudaSuccess) { \
        printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(status), __FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

template<unsigned int n>
__global__ void getClosestCenters(const float* objects, int N, int k, const float* centers, int* membership, int* delta)
{
    auto objectId = blockDim.x * blockIdx.x + threadIdx.x;
    if(objectId >= N)
    {
        return;
    }
    const float* object = objects + (objectId * n);
    
    auto closestCenterIndex = getClosestCenterIndex<n>(object, centers, k);

    if (membership[objectId] != closestCenterIndex)
    {
        // if the center has changed, increment delta
        membership[objectId] = closestCenterIndex;
        atomicAdd(delta, 1);
    }
}

template <unsigned int n>
int* kmeansGpu(const float* objects, int N, int k, float** centersOutput, float threshold)
{
    int gridSize = (float)N / BLOCK_SIZE + 1; // we want to have enough threads to cover the whole objects array (one thread -> one object)
    auto delta = 0;
    float* centers = new float[k * n];
    auto membership = new int[N];
    float* newCenters = new float[k * n];
    int* newClusterSizes = new int[k];

    float* dev_objects = 0;
    int* dev_delta = 0;
    float* dev_centers = 0;
    int* dev_membership = 0;
    ERR(cudaMalloc(&dev_objects, N * n * sizeof(float)));
    ERR(cudaMalloc(&dev_centers, k * n * sizeof(float)));
    ERR(cudaMalloc(&dev_membership, N * sizeof(int)));
    ERR(cudaMalloc(&dev_delta, sizeof(int)));
    ERR(cudaMemcpy(dev_objects, objects, N * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_membership, NO_MEMBERSHIP, N * sizeof(int)));

    memset(membership, NO_MEMBERSHIP, N * sizeof(int));
    // initialize cluster centers as first k objects
    memcpy(centers, objects, k * n * sizeof(float));

    do {
        ERR(cudaMemset(dev_delta, 0, sizeof(int)));
        ERR(cudaMemcpy(dev_centers, centers, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
        getClosestCenters<n><<<gridSize, BLOCK_SIZE>>>(dev_objects, N, k, dev_centers, dev_membership, dev_delta);
        ERR(cudaMemcpy(&delta, dev_delta, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ERR(cudaMemcpy(membership, dev_membership, N * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        memset(newCenters, 0, k * n * sizeof(float));
        memset(newClusterSizes, 0, k * sizeof(int));

        for (int i = 0; i < N; ++i)
        {
            auto object = objects + (i * n);
            // calculate sum of all cluster members
            ++newClusterSizes[membership[i]];
            for(int l = 0; l < n; ++l)
            {
                newCenters[membership[i] * n + l] += object[l];
            }
        }

        //std::cout << "Delta " << delta << " N " << N << std::endl;

        // calculate new cluster centers as averages of cluster members
        for(int i = 0; i < k; ++i)
        {
            if (newClusterSizes[i] == 0)
            {
                continue;
            }
            for(int j = 0; j < n; ++j)
            {
                centers[i * n + j] = newCenters[i * n + j] / newClusterSizes[i];
                //std::cout << centers[i * n + j] << ' ';
            }
            //std::cout << std::endl;
        }
        std::cout << "End of an iteration with delta " << delta << std::endl;
    } while ((float)delta / N > threshold);

    ERR(cudaFree(dev_objects));
    ERR(cudaFree(dev_centers));
    ERR(cudaFree(dev_membership));
    ERR(cudaFree(dev_delta));
    delete[] newCenters;
    delete[] newClusterSizes;
    *centersOutput = centers;
    return membership;
}
