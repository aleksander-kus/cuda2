#define NO_MEMBERSHIP -1

template <unsigned int n>
float distanceSquared(const float* object1, const float* object2)
{
    float sum = 0;
    for(int i = 0; i < n; ++i)
    {
        sum += (object1[i] - object2[i]) * (object1[i] - object2[i]);
    }
    return sum;
}

template <unsigned int n>
int* kmeansCpu(const float* objects, int N, int k, float** centersOutput, float threshold)
{
    auto delta = 0;
    float* centers = new float[k * n];
    auto membership = new int[N];
    float* newCenters = new float[k*n];
    int* newClusterSizes = new int[k];

    memset(membership, NO_MEMBERSHIP, N * sizeof(int));
    // initialize cluster centers as first k objects
    memcpy(centers, objects, k * n * sizeof(float));

    do {
        delta = 0;
        memset(newCenters, 0, k * n * sizeof(float));
        memset(newClusterSizes, 0, k * sizeof(int));
        for (int i = 0; i < N; ++i)
        {
            const float* object = objects + (i * n);
            // find the closest center
            float minDistance = INT_MAX;
            int minDistanceIndex = -1;
            for (int j = 0; j < k; ++j)
            {
                // calculate distance to center with index j
                auto dist = distanceSquared<n>(object, centers + (j * n));
                if(dist < minDistance)
                {
                    minDistance = dist;
                    minDistanceIndex = j;
                }
            }
            if (membership[i] != minDistanceIndex)
            {
                // if the center has changed, increment delta
                membership[i] = minDistanceIndex;
                ++delta;
            }

            // calculate sum of all cluster members
            ++newClusterSizes[membership[i]];
            for(int l = 0; l < n; ++l)
            {
                newCenters[membership[i] * n + l] += object[l];
            }
        }

        //std::cout << "Delta " << delta << " N " << N << std::endl;

        // if there were sufficiently few changes, stop
        if((float)delta/N <= threshold)
        {
            break;
        }

        // calculate new cluster centers as averages of cluster members
        for(int i = 0; i < k; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                centers[i * n + j] = newCenters[i * n + j] / newClusterSizes[i];
                //std::cout << centers[i * n + j] << ' ';
            }
            //std::cout << std::endl;
        }
        //std::cout << "End of an iteration" << std::endl;
    } while(true);

    delete[] newCenters;
    delete[] newClusterSizes;
    *centersOutput = centers;
    return membership;
}
