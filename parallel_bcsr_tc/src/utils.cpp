#include "../include/utils.hpp"

void printMatrix(int *matrix, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            std::cout<<matrix[i*n + j]<<" ";
        }
        std::cout<<std::endl;
    }
}

void printArray(int *array, int n){
    for (int i = 0; i < n; i++){
        std::cout<<array[i]<<" ";
    }
    std::cout<<std::endl;
}

void printStatus(int *capacity, int *excess, int *height, int *residual, int *totalExcess, int n){
    std::cout<<"Capacity"<<std::endl;
    printMatrix(capacity, n);
    std::cout<<"Excess"<<std::endl;
    printArray(excess, n);
    std::cout<<"Height"<<std::endl;
    printArray(height, n);
    std::cout<<"Residual"<<std::endl;
    printMatrix(residual, n);
    std::cout<<"Total Excess: "<<*totalExcess<<std::endl;
}