
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>

// bool solveBacktracking(char* board)
// {
//     int i = 0, j = 0;

//     if (!findEmpty(board, i, j))
//     {
//         return true;
//     }

//     for(int num = 1; num < 10; ++num)
//     {
//         if (tryToInsert(board, i, j, num))
//         {
//             board[i * BOARDSIZE + j] = num;
//             if (solveBacktracking(board))
//             {
//                 return true;
//             }
//             board[i * BOARDSIZE + j] = BLANK;
//         }
//     }
//     return false;
// }

// char* solveCpu(const char* board)
// {
//     char* copy = new char[BOARDLENGTH];
//     memcpy(copy, board, sizeof(char) * BOARDLENGTH);
//     auto result = solveBacktracking(copy);
//     if (result)
//     {
//         return copy;
//     }
//     else
//     {
//         free(copy);
//         return nullptr;
//     }
// }

// template <unsigned int n>
// int* kmeansCpu(const float* objects, int N, int k)
// {
//     for(int i = 0; i < N; ++i)
//     {
//         for(int j = 0; j < n; ++j)
//         {
//             std::cout << objects[n * i + j] << ' ';
//         }
//         std::cout << std::endl;
//     }
//     return NULL;
// }