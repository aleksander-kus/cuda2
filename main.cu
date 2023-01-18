#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <bitset>
#include <getopt.h>
#include <sstream>

#include "kmeanscpu.cuh"
#include "kmeansgpu.cuh"

#define DIM 2

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

// void printSudoku(char* board)
// {
//     const std::string lineBreak = "+-------+-------+-------+\n";
//     const std::string columnBreak = "| ";

//     for (auto i = 0; i < BOARDSIZE; ++i)
//     {
//         if (i % 3 == 0)
//         {
//             std::cout << lineBreak;
//         }
//         for (auto j = 0; j < BOARDSIZE; ++j)
//         {
//             if (j % 3 == 0)
//             {
//                 std::cout << columnBreak;
//             }

//             auto value = board[i * BOARDSIZE + j];
//             if (value == BLANK)
//             {
//                 std::cout << ". ";
//             }
//             else
//             {
//                 std::cout << (int)value << ' ';
//             }
//         }
//         std::cout << columnBreak << std::endl;
//     }
//     std::cout << lineBreak;
// }

// bool checkIfSudokuValid(const char* board)
// {
//     std::bitset<10> bitset;
//     // check rows
//     for (int i = 0; i < BOARDSIZE; ++i)
//     {
//         for (int j = 0; j < BOARDSIZE; ++j)
//         {
//             auto value = board[i * BOARDSIZE + j];
//             if (value == BLANK)
//             {
//                 continue;
//             }
//             if (bitset.test(value))
//             {
//                 return false;
//             }
//             bitset.set(value, true);
//         }
//         bitset.reset();
//     }

//     // check columns
//     for (int j = 0; j < BOARDSIZE; ++j)
//     {
//         for (int i = 0; i < BOARDSIZE; ++i)
//         {
//             auto value = board[i * BOARDSIZE + j];
//             if (value == BLANK)
//             {
//                 continue;
//             }
//             if (bitset.test(value))
//             {
//                 return false;
//             }
//             bitset.set(value, true);
//         }
//         bitset.reset();
//     }

//     // check boxes
//     for (int i = 0; i < 3; ++i)
//     {
//         for (int j = 0; j < 3; ++j)
//         {
//             int rowCenter = (i / 3) * 3 + 1;
//             int columnCenter = (j / 3) * 3 + 1;
//             for (int k = -1; k < 2; ++k)
//             {
//                 for (int l = -1; l < 2; ++l)
//                 {
//                     auto value = board[(rowCenter + k) * BOARDSIZE + (columnCenter + l)];
//                     if (value == BLANK)
//                     {
//                         continue;
//                     }
//                     if (bitset.test(value))
//                     {
//                         return false;
//                     }
//                     bitset.set(value, true);
//                 }
//             }
//             bitset.reset();
//         }
//     }

//     return true;
// }

// void compareResults(char* cpu, char* gpu)
// {
//     bool ok = true;
//     for (int i = 0; i < BOARDSIZE; ++i)
//     {
//         for (int j = 0; j < BOARDSIZE; ++j)
//         {
//             if (cpu[i * BOARDSIZE + j] != gpu[i * BOARDSIZE + j])
//             {
//                 ok = false;
//             }
//         }
//     }
//     if (ok)
//     {
//         std::cout << "Cpu and gpu results match" << std::endl;
//     }
//     else
//     {
//         std::cout << "Cpu and gpu results don't match" << std::endl;
//     }
// }

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
    
    int N = 0; // number of objects
    int k = 3;
    auto objects = readObjectsFromFile<DIM>(filepath, &N);

    float* cpuCenters, *gpuCenters;
    int* cpuMembership, *gpuMembership;
    if (!isGpuOnly)
    {
        cpuMembership = kmeansCpu<DIM>(objects, N, k, &cpuCenters);

        writeResultsToFile<DIM>("results/cpu.membership", "results/cpu.centers", N, k, cpuMembership, cpuCenters);

        for (int i = 0; i < k; ++i)
        {
            std::cout << "CLUSTER " << i << std::endl;
            std::cout << "MEMBERS" << std::endl;
            for (int j = 0; j < N; ++j)
            {
                if (cpuMembership[j] != i)
                    continue;
                for (int l = 0; l < DIM; ++l)
                {
                    std::cout << objects[j * DIM + l] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "CENTER IS " << cpuCenters[i * DIM] << ' ' << cpuCenters[i * DIM + 1] << std::endl;
            std::cout << std::endl;
        }
    }

    if (!isCpuOnly)
    {
        gpuMembership = kmeansCpu<DIM>(objects, N, k, &gpuCenters);

        writeResultsToFile<DIM>("results/cpu.membership", "results/cpu.centers", N, k, gpuMembership, gpuCenters);

        for (int i = 0; i < k; ++i)
        {
            std::cout << "CLUSTER " << i << std::endl;
            std::cout << "MEMBERS" << std::endl;
            for (int j = 0; j < N; ++j)
            {
                if (gpuMembership[j] != i)
                    continue;
                for (int l = 0; l < DIM; ++l)
                {
                    std::cout << objects[j * DIM + l] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "CENTER IS " << gpuCenters[i * DIM] << ' ' << gpuCenters[i * DIM + 1] << std::endl;
            std::cout << std::endl;
        }
    }

    delete[] objects;
    delete[] cpuMembership;
    delete[] cpuCenters;
    delete[] gpuMembership;
    delete[] gpuCenters;
    return 0;
}
