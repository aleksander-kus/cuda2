#include "cuda_runtime.h"

#include <iostream>
#include <chrono>

#include "kmeanscpu.cuh"

#define GRID_SIZE 1024
#define BLOCK_SIZE 1024

#define ERR(status) { \
    if (status != cudaSuccess) { \
        printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(status), __FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

template<unsigned int n>
__global__ void getClosestCenters(const float* objects, int N, int k, const float* centers, int* membership, bool* membershipChanged)
{
    auto objectId = blockDim.x * blockIdx.x + threadIdx.x;
    while (objectId < N)
    {
        const float* object = objects + (objectId * n);
        
        auto closestCenterIndex = getClosestCenterIndex<n>(object, centers, k);

        if (membership[objectId] != closestCenterIndex)
        {
            // if the center has changed, increment delta
            membership[objectId] = closestCenterIndex;
            membershipChanged[objectId] = true;
        }

        objectId += gridDim.x + blockDim.x;
    }
}

template<unsigned int n>
__global__ void calculateNewCenters(float* centers, const float* clusterSum, const int* clusterCount, int k)
{
    auto objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if(objectId >= k || clusterCount[objectId] == 0)
    {
        return;
    }

    for(int i = 0; i < n; ++i)
    {
        centers[objectId * n + i] = clusterSum[objectId * n + i] / clusterCount[objectId];
    }
}

template <unsigned int n>
int* kmeansGpu(const float* objects, int N, int k, float** centersOutput, float threshold)
{
    int gridSize = (float)N / BLOCK_SIZE + 1; // we want to have enough threads to cover the whole objects array (one thread -> one object)
    auto delta = 0;
    auto membership = new int[N];
    auto membershipChanged = new bool[N];
    float* clusterSum = new float[k * n];
    int* clusterCount = new int[k];

    float* dev_objects = 0;
    //int* dev_delta = 0;
    float* dev_centers = 0;
    int* dev_membership = 0;
    bool* dev_membershipChanged = 0;
    float* dev_clusterSum = 0;
    int* dev_clusterCount = 0;
    ERR(cudaMalloc(&dev_objects, N * n * sizeof(float)));
    ERR(cudaMalloc(&dev_centers, k * n * sizeof(float)));
    ERR(cudaMalloc(&dev_membership, N * sizeof(int)));
    ERR(cudaMalloc(&dev_membershipChanged, N * sizeof(bool)));
    ERR(cudaMalloc(&dev_clusterSum, k * n * sizeof(float)));
    ERR(cudaMalloc(&dev_clusterCount, k * sizeof(int)));
    //ERR(cudaMalloc(&dev_delta, sizeof(int)));
    ERR(cudaMemcpy(dev_objects, objects, N * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_membership, NO_MEMBERSHIP, N * sizeof(int)));

    memset(membership, NO_MEMBERSHIP, N * sizeof(int));
    // initialize cluster centers as first k objects
    ERR(cudaMemcpy(dev_centers, objects, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    //memcpy(centers, objects, k * n * sizeof(float));

    do {
        delta = 0;
        //ERR(cudaMemset(dev_delta, 0, sizeof(int)));
        ERR(cudaMemset(dev_membershipChanged, 0, N * sizeof(bool)));
        //ERR(cudaMemcpy(dev_centers, centers, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
        getClosestCenters<n><<<gridSize, BLOCK_SIZE>>>(dev_objects, N, k, dev_centers, dev_membership, dev_membershipChanged);
        //ERR(cudaMemcpy(&delta, dev_delta, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ERR(cudaMemcpy(membership, dev_membership, N * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ERR(cudaMemcpy(membershipChanged, dev_membershipChanged, N * sizeof(bool), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        memset(clusterSum, 0, k * n * sizeof(float));
        memset(clusterCount, 0, k * sizeof(int));

        for (int i = 0; i < N; ++i)
        {
            if(membershipChanged[i])
                ++delta;
            auto object = objects + (i * n);
            // calculate sum of all cluster members
            ++clusterCount[membership[i]];
            for(int l = 0; l < n; ++l)
            {
                clusterSum[membership[i] * n + l] += object[l];
            }
        }
        ERR(cudaMemcpy(dev_clusterSum, clusterSum, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
        ERR(cudaMemcpy(dev_clusterCount, clusterCount, k * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));

        // calculate new cluster centers as averages of cluster members
        calculateNewCenters<n><<<(float)k / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_centers, dev_clusterSum, dev_clusterCount, k);
        std::cout << "End of an iteration with delta " << delta << std::endl;
    } while ((float)delta / N > threshold);

    float* centers = new float[k * n];
    ERR(cudaMemcpy(centers, dev_centers, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost))
    ERR(cudaFree(dev_objects));
    ERR(cudaFree(dev_centers));
    ERR(cudaFree(dev_membership));
    ERR(cudaFree(dev_membershipChanged));
    ERR(cudaFree(dev_clusterSum));
    ERR(cudaFree(dev_clusterCount));
    //ERR(cudaFree(dev_delta));
    delete[] membershipChanged;
    delete[] clusterSum;
    delete[] clusterCount;
    
    *centersOutput = centers;
    return membership;
}
