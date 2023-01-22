#include "kmeanscpu.cuh"

#include "cuda_runtime.h"

#define BLOCK_SIZE 1024

#define ERR(status) { \
    if (status != cudaSuccess) { \
        printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(status), __FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

template<unsigned int n>
__global__ void getClosestCenters(const float* objects, int N, int k, const float* centers, int* membership, int* membershipChanged)
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
        membershipChanged[objectId] = 1;
    }
}

template<unsigned int n>
__global__ void calculateDelta(const int *membershipChanged, int arraySize, int *result) {
    __shared__ int shared[BLOCK_SIZE];
    const int gridSize = BLOCK_SIZE*gridDim.x;
    int threadId = threadIdx.x;
    int globalThreadId = threadId + blockIdx.x*BLOCK_SIZE;

    auto sum = 0;
    for (int i = globalThreadId; i < arraySize; i += gridSize)
        sum += membershipChanged[i];
    shared[threadId] = sum;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1) 
    {
        if (threadId<i)
        {
            shared[threadId] += shared[threadId+i];
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        result[blockIdx.x] = shared[0];
    }
}


template<unsigned int n>
__global__ void calculateNewCenters(float* centers, const float* clusterSum, const int* clusterCount, int k)
{
    auto clusterId = blockDim.x * blockIdx.x + threadIdx.x;

    if(clusterId >= k || clusterCount[clusterId] == 0)
    {
        return;
    }

    for(int i = 0; i < n; ++i)
    {
        centers[clusterId * n + i] = clusterSum[clusterId * n + i] / clusterCount[clusterId];
    }
}

template <unsigned int n>
int* kmeansGpu(const float* objects, int N, int k, float** centersOutput, bool isDebug, float threshold)
{
    int gridSize = (float)N / BLOCK_SIZE + 1; // we want to have enough threads to cover the whole objects array (one thread -> one object)
    auto delta = 0;
    auto membership = new int[N];
    float* clusterSum = new float[k * n];
    int* clusterCount = new int[k];

    float* dev_objects = 0;
    float* dev_centers = 0;
    int* dev_membership = 0;
    int* dev_membershipChanged = 0;
    float* dev_clusterSum = 0;
    int* dev_clusterCount = 0;
    int* dev_deltaSum = 0;
    ERR(cudaMalloc(&dev_objects, N * n * sizeof(float)));
    ERR(cudaMalloc(&dev_centers, k * n * sizeof(float)));
    ERR(cudaMalloc(&dev_membership, N * sizeof(int)));
    ERR(cudaMalloc(&dev_membershipChanged, N * sizeof(int)));
    ERR(cudaMalloc(&dev_clusterSum, k * n * sizeof(float)));
    ERR(cudaMalloc(&dev_clusterCount, k * sizeof(int)));
    ERR(cudaMalloc(&dev_deltaSum, gridSize * sizeof(int)));
    ERR(cudaMemcpy(dev_objects, objects, N * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_membership, NO_MEMBERSHIP, N * sizeof(int)));

    memset(membership, NO_MEMBERSHIP, N * sizeof(int));
    // initialize cluster centers as first k objects
    ERR(cudaMemcpy(dev_centers, objects, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));

    do {
        ERR(cudaMemset(dev_membershipChanged, 0, N * sizeof(int)));
        ERR(cudaMemset(dev_deltaSum, 0, gridSize * sizeof(int)));
        getClosestCenters<n><<<gridSize, BLOCK_SIZE>>>(dev_objects, N, k, dev_centers, dev_membership, dev_membershipChanged);
        ERR(cudaMemcpy(membership, dev_membership, N * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

        calculateDelta<n><<<gridSize, BLOCK_SIZE>>>(dev_membershipChanged, N, dev_deltaSum);
        calculateDelta<n><<<1, BLOCK_SIZE>>>(dev_deltaSum, gridSize, dev_deltaSum);
        ERR(cudaMemcpy(&delta, dev_deltaSum, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        memset(clusterSum, 0, k * n * sizeof(float));
        memset(clusterCount, 0, k * sizeof(int));

        for (int i = 0; i < N; ++i)
        {
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
        if (isDebug)
        {
            std::cout << "End of an iteration with delta " << delta << std::endl;
        }
    } while ((float)delta / N > threshold);

    float* centers = new float[k * n];
    ERR(cudaMemcpy(centers, dev_centers, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost))
    ERR(cudaFree(dev_objects));
    ERR(cudaFree(dev_centers));
    ERR(cudaFree(dev_membership));
    ERR(cudaFree(dev_membershipChanged));
    ERR(cudaFree(dev_clusterSum));
    ERR(cudaFree(dev_clusterCount));
    ERR(cudaFree(dev_deltaSum));
    delete[] clusterSum;
    delete[] clusterCount;
    
    *centersOutput = centers;
    return membership;
}

template<unsigned int n>
__global__ void sumClusters(const float* objects, const int* membership, float* clusterSum, int* clusterCount, int N, int k)
{
    auto objectId = blockDim.x * blockIdx.x + threadIdx.x;
    if(objectId >= N)
    {
        return;
    }
    auto object = objects + (objectId * n);
    auto clusterId = membership[objectId];

    atomicAdd(&clusterCount[clusterId], 1);
    for(int i = 0; i < n; ++i)
    {
        atomicAdd(&clusterSum[clusterId * n + i], object[i]);
    }
}

template <unsigned int n>
int* kmeansGpu2(const float* objects, int N, int k, float** centersOutput, bool isDebug, float threshold)
{
    int gridSize = (float)N / BLOCK_SIZE + 1; // we want to have enough threads to cover the whole objects array (one thread -> one object)
    auto delta = 0;
    auto membership = new int[N];

    float* dev_objects = 0;
    float* dev_centers = 0;
    int* dev_membership = 0;
    int* dev_membershipChanged = 0;
    float* dev_clusterSum = 0;
    int* dev_clusterCount = 0;
    int* dev_deltaSum = 0;
    ERR(cudaMalloc(&dev_objects, N * n * sizeof(float)));
    ERR(cudaMalloc(&dev_centers, k * n * sizeof(float)));
    ERR(cudaMalloc(&dev_membership, N * sizeof(int)));
    ERR(cudaMalloc(&dev_membershipChanged, N * sizeof(int)));
    ERR(cudaMalloc(&dev_clusterSum, k * n * sizeof(float)));
    ERR(cudaMalloc(&dev_clusterCount, k * sizeof(int)));
    ERR(cudaMalloc(&dev_deltaSum, gridSize * sizeof(int)));
    ERR(cudaMemcpy(dev_objects, objects, N * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_membership, NO_MEMBERSHIP, N * sizeof(int)));

    memset(membership, NO_MEMBERSHIP, N * sizeof(int));
    // initialize cluster centers as first k objects
    ERR(cudaMemcpy(dev_centers, objects, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));

    do {
        ERR(cudaMemset(dev_membershipChanged, 0, N * sizeof(int)));
        ERR(cudaMemset(dev_deltaSum, 0, gridSize * sizeof(int)));
        getClosestCenters<n><<<gridSize, BLOCK_SIZE>>>(dev_objects, N, k, dev_centers, dev_membership, dev_membershipChanged);
        ERR(cudaMemcpy(membership, dev_membership, N * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

        calculateDelta<n><<<gridSize, BLOCK_SIZE>>>(dev_membershipChanged, N, dev_deltaSum);
        calculateDelta<n><<<1, BLOCK_SIZE>>>(dev_deltaSum, gridSize, dev_deltaSum);
        ERR(cudaMemcpy(&delta, dev_deltaSum, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ERR(cudaMemset(dev_clusterSum, 0, k * n * sizeof(float)));
        ERR(cudaMemset(dev_clusterCount, 0, k * sizeof(float)));
        sumClusters<n><<<gridSize, BLOCK_SIZE>>>(dev_objects, dev_membership, dev_clusterSum, dev_clusterCount, N, k);

        // calculate new cluster centers as averages of cluster members
        calculateNewCenters<n><<<(float)k / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_centers, dev_clusterSum, dev_clusterCount, k);
        if (isDebug)
        {
            std::cout << "End of an iteration with delta " << delta << std::endl;
        }
    } while ((float)delta / N > threshold);

    float* centers = new float[k * n];
    ERR(cudaMemcpy(centers, dev_centers, k * n * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost))
    ERR(cudaFree(dev_objects));
    ERR(cudaFree(dev_centers));
    ERR(cudaFree(dev_membership));
    ERR(cudaFree(dev_membershipChanged));
    ERR(cudaFree(dev_clusterSum));
    ERR(cudaFree(dev_clusterCount));
    ERR(cudaFree(dev_deltaSum));
    
    *centersOutput = centers;
    return membership;
}
