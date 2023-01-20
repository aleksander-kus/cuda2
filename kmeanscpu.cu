#define NO_MEMBERSHIP -1

#include <iostream>

template <unsigned int n>
__host__ __device__ inline float distanceSquared(const float* object1, const float* object2)
{
    float sum = 0;
    for(int i = 0; i < n; ++i)
    {
        sum += (object1[i] - object2[i]) * (object1[i] - object2[i]);
    }
    return sum;
}

template <unsigned int n>
__host__ __device__ inline int getClosestCenterIndex(const float* object, const float* centers, int k)
{
    float minDistance = __FLT_MAX__;
    int minIndex = -1;
    for (int j = 0; j < k; ++j)
    {
        // calculate distance to center with index j
        auto dist = distanceSquared<n>(object, centers + (j * n));
        if (dist < minDistance) // x^2 is monotonous on <0, +inf) so we can compare squared distances
        {
            minDistance = dist;
            minIndex = j;
        }
    }
    return minIndex;
}

template <unsigned int n>
int* kmeansCpu(const float* objects, int N, int k, float** centersOutput, float threshold)
{
    auto delta = 0;
    float* centers = new float[k * n];
    auto membership = new int[N];
    float* clusterSum = new float[k * n];
    int* clusterCount = new int[k];

    memset(membership, NO_MEMBERSHIP, N * sizeof(int));
    // initialize cluster centers as first k objects
    memcpy(centers, objects, k * n * sizeof(float));

    do {
        delta = 0;
        memset(clusterSum, 0, k * n * sizeof(float));
        memset(clusterCount, 0, k * sizeof(int));
        for (int i = 0; i < N; ++i)
        {
            const float* object = objects + (i * n);
            // find the closest center
            auto closestCenterIndex = getClosestCenterIndex<n>(object, centers, k);

            if (membership[i] != closestCenterIndex)
            {
                // if the center has changed, increment delta
                membership[i] = closestCenterIndex;
                ++delta;
            }

            // calculate sum of all cluster members
            ++clusterCount[membership[i]];
            for(int l = 0; l < n; ++l)
            {
                clusterSum[membership[i] * n + l] += object[l];
            }
        }

        //std::cout << "Delta " << delta << " N " << N << std::endl;

        // calculate new cluster centers as averages of cluster members
        for (int i = 0; i < k; ++i)
        {
            if (clusterCount[i] == 0)
            {
                continue;
            }
            for (int j = 0; j < n; ++j)
            {
                centers[i * n + j] = clusterSum[i * n + j] / clusterCount[i];
                //std::cout << centers[i * n + j] << ' ';
            }
            //std::cout << std::endl;
        }
        std::cout << "End of an iteration with delta " << delta << std::endl;
    } while ((float)delta / N > threshold);

    delete[] clusterSum;
    delete[] clusterCount;
    *centersOutput = centers;
    return membership;
}
