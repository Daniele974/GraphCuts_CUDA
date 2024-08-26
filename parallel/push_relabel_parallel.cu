#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <list>

using namespace std;

#define INF 1000000

#define _DEBUG 0

void printMatrix(int *matrix, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            cout<<matrix[i*n + j]<<" ";
        }
        cout<<endl;
    }
}

void printArray(int *array, int n){
    for (int i = 0; i < n; i++){
        cout<<array[i]<<" ";
    }
    cout<<endl;
}

void printStatus(int *capacity, int *excess, int *height, int *residual, int *totalExcess, int n){
    cout<<"Capacity"<<endl;
    printMatrix(capacity, n);
    cout<<"Excess"<<endl;
    printArray(excess, n);
    cout<<"Height"<<endl;
    printArray(height, n);
    cout<<"Residual"<<endl;
    printMatrix(residual, n);
    cout<<"Total Excess: "<<*totalExcess<<endl;
}

void initialize(int *capacity, int *excess, int *height, int *residual, int *totalExcess, int n, int s){
    for (int i = 0; i < n; i++){
        height[i] = 0;
        excess[i] = 0;
    }
    
    height[s] = n;
    *totalExcess = 0;

    // Inizializzazione residui
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            residual[i*n + j] = capacity[i*n + j];
        }
    }

    // Calcolo del preflusso
    for (int i = 0; i < n; i++){
        if (capacity[s*n + i] > 0){
            residual[s*n + i] = residual[s*n + i] - capacity[s*n + i];
            residual[i*n + s] = residual[i*n + s] + capacity[s*n + i];
            excess[i] = capacity[s*n + i];
            *totalExcess = *totalExcess + capacity[s*n + i];
        }
    }
}

__global__ void pushKernel(int *d_capacity, int *d_excess, int *d_height, int *d_residual, int n){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < n && u != 0 && u != n-1){
        int cycle = n; //KERNEL_CYCLES
        int e1, h1, h2, v1, delta;

        while (cycle > 0){
            if(d_excess[u] > 0 && d_height[u] < n){
                e1 = d_excess[u];
                h1 = INF;
                v1 = NULL;
                for (int v = 0; v < n; v++){
                    if(d_residual[u*n + v] > 0){
                        h2 = d_height[v];
                        if(h2 < h1){
                            v1 = v;
                            h1 = h2;
                        }
                    }                
                }
                if(d_height[u] > h1){
                    delta = min(e1, d_residual[u*n + v1]);
                    atomicAdd(&d_residual[v1*n + u], delta);
                    atomicSub(&d_residual[u*n + v1], delta);
                    atomicAdd(&d_excess[v1], delta);
                    atomicSub(&d_excess[u], delta);
                }
                else{
                    d_height[u] = h1 + 1;
                }
            }
            cycle = cycle - 1;
        }
    }
}

void globalRelabel(int *capacity, int *excess, int *height, int *residual, int *totalExcess, bool *scanned, bool *mark, int n, int t){
    for (int u = 0; u < n; u++){
        for (int v = 0; v < n; v++){
            if(capacity[u*n + v] > 0){
                if(height[u] > height[v] + 1){
                    excess[u] = excess[u] - residual[u*n + v];
                    excess[v] = excess[v] + residual[u*n + v];
                    residual[v*n + u] = residual[v*n + u] + residual[u*n + v];
                    residual[u*n + v] = 0;
                }
            }
        }
    }
    
    list<int> queue;
    int x, y, current;
    
    for (int i = 0; i < n; i++){
        scanned[i] = false;
    }
    queue.push_back(t);
    scanned[t] = true;
    height[t] = 0;
    while (!queue.empty()){
        x = queue.front();
        queue.pop_front();
        current = height[x] +1;
        for(y = 0; y < n; y++){
            if(residual[y*n + x] > 0 && !scanned[y]){
                height[y] = current;
                scanned[y] = true;
                queue.push_back(y);
            }
        }
    }
    
    bool allScanned = true;
    for(int i = 0; i < n; i++){
        if(scanned[i]==false){
            allScanned = false;
            break;
        }
    }
    if(allScanned==false){
        for (int i = 0; i < n; i++){
            if(!(scanned[i] || mark[i])){
                mark[i] = true;
                *totalExcess = *totalExcess - excess[i];
            }
        }
    }
}

int pushRelabel(int *capacity, int *excess, int *height, int *residual, int *d_capacity, int *d_excess, int *d_height, int *d_residual, int *totalExcess, int n, int s, int t){
    bool *scanned = (bool*)malloc(n*sizeof(bool));
    bool *mark = (bool*)malloc(n*sizeof(bool));
    for (int i = 0; i < n; i++){
        mark[i] = false;
    }

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    while ((excess[s]+excess[t])<*totalExcess){
        
        if(_DEBUG) cout << "Push..." << endl;
        cudaMemcpy(d_height, height, n*sizeof(int), cudaMemcpyHostToDevice);
        pushKernel<<<gridSize, blockSize>>>(d_capacity, d_excess, d_height, d_residual, n);
        cudaDeviceSynchronize();
        if(_DEBUG) cout << "Push done" << endl;

        cudaMemcpy(excess, d_excess, n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(height, d_height, n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(residual, d_residual, n*n*sizeof(int), cudaMemcpyDeviceToHost);

        if(_DEBUG) printStatus(capacity, excess, height, residual, totalExcess, n);

        if(_DEBUG) cout << "Global relabel..." << endl;
        globalRelabel(capacity, excess, height, residual, totalExcess, scanned, mark, n, t);
        if(_DEBUG) cout << "Global relabel done" << endl;
        
        if(_DEBUG) printStatus(capacity, excess, height, residual, totalExcess, n);
        
    }
    return excess[t];
}

int main(int argc, char const *argv[]){
    int n = 4;
    int s = 0;
    int t = n-1;
    int *capacity = (int *)malloc(n*n*sizeof(int));
    int *flow = (int *)malloc(n*n*sizeof(int));
    int *excess = (int *)malloc(n*sizeof(int));
    int *height = (int *)malloc(n*sizeof(int));
    int *residual = (int *)malloc(n*n*sizeof(int));
    int *totalExcess = (int *)malloc(sizeof(int));

    capacity[0] = 0;
    capacity[1] = 2;
    capacity[2] = 4;
    capacity[3] = 0;
    capacity[4] = 0;
    capacity[5] = 0;
    capacity[6] = 3;
    capacity[7] = 1;
    capacity[8] = 0;
    capacity[9] = 0;
    capacity[10] = 0;
    capacity[11] = 5;
    capacity[12] = 0;
    capacity[13] = 0;
    capacity[14] = 0;
    capacity[15] = 0;


    int *d_capacity, *d_excess, *d_height, *d_residual;    

    cudaMalloc((void**)&d_capacity, n*n*sizeof(int));
    cudaMalloc((void**)&d_excess, n*sizeof(int));
    cudaMalloc((void**)&d_height, n*sizeof(int));
    cudaMalloc((void**)&d_residual, n*n*sizeof(int));

    initialize(capacity, excess, height, residual, totalExcess, n, s);

    if(_DEBUG) cout << "Initialization done" << endl;
    if(_DEBUG) printStatus(capacity, excess, height, residual, totalExcess, n);
    
    cudaMemcpy(d_capacity, capacity, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_excess, excess, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_height, height, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, residual, n*n*sizeof(int), cudaMemcpyHostToDevice);
    
    if(_DEBUG) cout << "Push relabel..." << endl;
    int maxFlow = pushRelabel(capacity, excess, height, residual, d_capacity, d_excess, d_height, d_residual, totalExcess, n, s, t);
    if(_DEBUG) cout << "Push relabel done" << endl;
    cout<<"Max flow: "<<maxFlow<<endl;


    cudaFree(d_capacity);
    cudaFree(d_excess);
    cudaFree(d_height);
    cudaFree(d_residual);

    free(capacity);
    free(flow);
    free(excess);
    free(height);
    free(residual);

    return 0;
}
