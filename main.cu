#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <bitset>
#include <getopt.h>
#include <sstream>
#include <cstdlib>

#include "kmeanscpu.cuh"
#include "kmeansgpu.cuh"

#define SMALL_DIM 2

template<unsigned int n>
float* readObjectsFromFile(std::string filepath, int* N)
{
    std::ifstream fileStream(filepath);
    if(!fileStream.good())
    {
        return NULL;
    }
    std::string input, number;
    getline(fileStream, input);
    *N = stoi(input);
    auto objects = new float[n * (*N)];
    int index = 0;
    while (getline(fileStream, input))
    {
        std::istringstream stream(input);
        while(getline(stream, number, ' ')) {
            objects[index++] = stof(number);
        }
    }
    return objects;
}

template<unsigned int n>
void writeResultsToFile(const char* membershipFilePath, const char* centersFilePath, int N, int k, int* membership, float* centers)
{
    std::ofstream membershipFileStream(membershipFilePath, std::ofstream::out | std::ofstream::trunc);
    if(!membershipFileStream.good())
    {
        std::cout << "kekium";
        return;
    }
    for(int i = 0; i < N; ++i)
    {
        membershipFileStream << membership[i] << std::endl;
    }
    membershipFileStream.close();
    std::ofstream centersFileStream(centersFilePath, std::ofstream::out | std::ofstream::trunc);
    if(!centersFileStream.good())
    {
        return;
    }
    for(int i = 0; i < k; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            centersFileStream << centers[i * n + j] << " ";
        }
        centersFileStream << std::endl;
    }
    centersFileStream.close();
}

template<unsigned int n>
float* generateRandomData(int N)
{
    srand(1234);

    float* data = new float[N * n];
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            data[i * n + j] = -1000 + (rand() % 2000);
        }
    }

    return data;
}

void usage()
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  kmeans.out [options] filepath" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --cpu-only      only run cpu algorithm" << std::endl;
    std::cout << "  -g, --gpu-only      only run gpu algorithm" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    int c;
    bool isCpuOnly = false;
    bool isGpuOnly = false;
    static struct option long_options[] = {
        {"cpu-only", no_argument, NULL, 'c'},
        {"gpu-only", no_argument, NULL, 'g'},
        { NULL, 0, NULL, 0 }
    };

    while (1)
    {
        c = getopt_long(argc, argv, "cg", long_options, NULL);
        if(c == -1)
            break;

        switch(c)
        {
            case 'c':
                isCpuOnly = true;
                break;
            case 'g':
                isGpuOnly = true;
                break;
            default:
                usage();
                break;
        }
    }

    if (optind != argc - 1) {
        usage();
    }

    std::string filepath = argv[optind++];

    if(isCpuOnly && isGpuOnly)
    {
        std::cout << "The -c and -g flags are mutually exclusive" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int N = 2000000; // number of objects
    int k = 10;
    //auto objects = readObjectsFromFile<SMALL_DIM>(filepath, &N);
    auto objects = generateRandomData<SMALL_DIM>(N);

    float* cpuCenters, *gpuCenters;
    int* cpuMembership, *gpuMembership;
    if (!isGpuOnly)
    {
        std::cout << "Solving kmeans cpu..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        cpuMembership = kmeansCpu<SMALL_DIM>(objects, N, k, &cpuCenters);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time for cpu: " << duration.count() << " microseconds" << std::endl;

        writeResultsToFile<SMALL_DIM>("results/cpu.membership", "results/cpu.centers", N, k, cpuMembership, cpuCenters);
    }

    if (!isCpuOnly)
    {
        std::cout << "Solving kmeans cpu..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        gpuMembership = kmeansGpu<SMALL_DIM>(objects, N, k, &gpuCenters);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time for cpu: " << duration.count() << " microseconds" << std::endl;

        writeResultsToFile<SMALL_DIM>("results/gpu.membership", "results/gpu.centers", N, k, gpuMembership, gpuCenters);
    }

    delete[] objects;
    delete[] cpuMembership;
    delete[] cpuCenters;
    delete[] gpuMembership;
    delete[] gpuCenters;
    return 0;
}
