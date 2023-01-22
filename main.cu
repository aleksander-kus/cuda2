#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <getopt.h>
#include <sstream>
#include <cstdlib>

#include "kmeansgpu.cuh"

#define DIM 3

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
float* generateRandomData(int N, int seed = 1234)
{
    srand(seed);

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

template<unsigned int n>
void checkResults(float* cpuCenters, float* gpuCenters, int* cpuMembership, int* gpuMembership, int N, int k)
{
    for (int i = 0; i < k * n; ++i)
    {
        if (cpuCenters[i] != gpuCenters[i])
        {
            std::cout << "Cluster centers do not match" << std::endl;
            return;
        }
    }

    for (int i = 0; i < N; ++i)
    {
        if (cpuMembership[i] != gpuMembership[i])
        {
            std::cout << "Cluster memberships do not match" << std::endl;
            return;
        }
    }

    std::cout << "Cpu and gpu results match" << std::endl;
}

void usage()
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  kmeans.out [-f filepath | -n N] [options] k " << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --cpu-only      only run cpu algorithm" << std::endl;
    std::cout << "  -f, --file          specify a path to a file with data" << std::endl;
    std::cout << "  -g, --gpu-only      only run gpu algorithm" << std::endl;
    std::cout << "  -n, --generate      generate a random set of N objects" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    int c;
    bool isCpuOnly = false;
    bool isGpuOnly = false;
    bool isFile = false;
    bool isGenerate = false;
    bool isDebug = false;
    static struct option long_options[] = {
        {"cpu-only", no_argument, NULL, 'c'},
        {"file", required_argument, NULL, 'f'},
        {"gpu-only", no_argument, NULL, 'g'},
        {"generate", required_argument, NULL, 'n'},
        {"debug", no_argument, NULL, 'd'},
        { NULL, 0, NULL, 0 }
    };
    std::string filepath;
    int N = 0;

    while (1)
    {
        c = getopt_long(argc, argv, "cdf:gn:", long_options, NULL);
        if(c == -1)
            break;

        switch(c)
        {
            case 'c':
                isCpuOnly = true;
                break;
            case 'd':
                isDebug = true;
                break;
            case 'f':
                isFile = true;
                filepath = optarg;
                break;
            case 'g':
                isGpuOnly = true;
                break;
            case 'n':
                isGenerate = true;
                N = atoi(optarg);
                if(N < 1)
                {
                    usage();
                }
                break;
            default:
                usage();
                break;
        }
    }

    if (optind != argc - 1 || (isFile && isGenerate) || !(isFile || isGenerate)) {
        usage();
    }
    int k = atoi(argv[optind++]);

    if (isCpuOnly && isGpuOnly)
    {
        std::cout << "The -c and -g flags are mutually exclusive" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // initialize data
    float* objects = 0;
    if (isFile)
    {
        objects = readObjectsFromFile<DIM>(filepath, &N);
    }
    else
    {
        objects = generateRandomData<DIM>(N);
    }

    float* cpuCenters, *gpuCenters;
    int* cpuMembership, *gpuMembership;
    if (!isGpuOnly)
    {
        std::cout << std::endl;
        std::cout << "Solving kmeans cpu..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        cpuMembership = kmeansCpu<DIM>(objects, N, k, &cpuCenters, isDebug);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time for cpu: " << duration.count() << " microseconds" << std::endl;

        std::cout << "Writing cpu results to files results/cpu.membership and results/cpu.centers" << std::endl;
        writeResultsToFile<DIM>("results/cpu.membership", "results/cpu.centers", N, k, cpuMembership, cpuCenters);
    }

    if (!isCpuOnly)
    {
        std::cout << std::endl;
        std::cout << "Solving kmeans gpu..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        gpuMembership = kmeansGpu<DIM>(objects, N, k, &gpuCenters, isDebug);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time for gpu: " << duration.count() << " microseconds" << std::endl;

        std::cout << "Writing gpu results to files results/gpu.membership and results/gpu.centers" << std::endl;
        writeResultsToFile<DIM>("results/gpu.membership", "results/gpu.centers", N, k, gpuMembership, gpuCenters);
    }

    if(!isCpuOnly && !isGpuOnly)
    {
        std::cout << std::endl;
        checkResults<DIM>(cpuCenters, gpuCenters, cpuMembership, gpuMembership, N, k);
    }

    std::cout << std::endl;
    std::cout << "Deleting objects" << std::endl;
    delete[] objects;
    if(!isGpuOnly)
    {
        delete[] cpuMembership;
        delete[] cpuCenters;
    }
    if(!isCpuOnly)
    {
        delete[] gpuMembership;
        delete[] gpuCenters;
    }
    return 0;
}
