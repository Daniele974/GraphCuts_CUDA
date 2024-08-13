#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace std;

#define INF 1000000


__global__ void initializeFlowExcessHeight(int *d_flow, int *d_excess, int *d_height, int n, int s){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        for(int i = 0; i < n; i++){
            // Ogni thread riempie una riga della matrice di flusso
            d_flow[tid*n + i] = 0;
        }
        // Inizializzazione dell'eccesso (INF se s, 0 altrimenti)
        d_excess[tid] = (tid == s) ? INF : 0;
        // Inizializzazione dell'altezza
        d_height[tid] = (tid == s) ? n : 0;
    }
}

__global__ void pushKernel(int *d_capacity, int *d_flow, int *d_excess, int *d_height, int n){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (u >= n) return;
    if (d_excess[u] <= 0) return;
    for (int v = 0; v < n; v++){
        if(d_capacity[u*n + v] - d_flow[u*n + v] > 0 && d_height[u] > d_height[v]){
            int delta = min(d_excess[u], d_capacity[u*n + v] - d_flow[u*n + v]);
            atomicSub(&d_excess[u], delta);
            atomicAdd(&d_excess[v], delta);
            atomicAdd(&d_flow[u*n + v], delta);
            atomicSub(&d_flow[v*n + u], delta);
        }
    }
}

__global__ void relabelKernel(int *d_capacity, int *d_flow, int *d_excess, int *d_height, int n){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (u >= n) return;

    if (d_excess[u] > 0){
        int minHeight = INF;
        for (int v = 0; v < n; v++){
            if (d_capacity[u*n + v] - d_flow[u*n + v] > 0){
                minHeight = min(minHeight, d_height[v]);
            }
        }
        if (minHeight < INF){
            d_height[u] = minHeight + 1;
        }
    }
}

int pushRelabel(int *capacity, int *flow, int *excess, int *height, int n, int s, int t){
    int *d_capacity, *d_flow, *d_excess, *d_height;
    cudaMalloc(&d_capacity, n*n*sizeof(int));
    cudaMalloc(&d_flow, n*n*sizeof(int));
    cudaMalloc(&d_excess, n*sizeof(int));
    cudaMalloc(&d_height, n*sizeof(int));

    cudaMemcpy(d_capacity, capacity, n*n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    // Inizializzazione flusso, eccesso e altezza
    initializeFlowExcessHeight<<<gridSize, blockSize>>>(d_flow, d_excess, d_height, n, s);

    // Esecuzione
    bool active = true;
    int count = 0;
    while (active){
        active = false;
        count++;
        // Push
        pushKernel<<<gridSize, blockSize>>>(d_capacity, d_flow, d_excess, d_height, n);
        cudaDeviceSynchronize();
        
        // Relabel
        relabelKernel<<<gridSize, blockSize>>>(d_capacity, d_flow, d_excess, d_height, n);
        cudaDeviceSynchronize();

        cudaMemcpy(excess, d_excess, n*sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n; i++){
            if (i!=s && i!=t && excess[i] > 0){
                cout<<"Sono qui"<<endl;
                active = true;
                break;
            }
        }
        
    }

    cudaMemcpy(flow, d_flow, n*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(height, d_height, n*n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_capacity);
    cudaFree(d_flow);
    cudaFree(d_excess);
    cudaFree(d_height);
    
    /* vector<int> minCutSet;
    for (int u = 0; u < n; u++){
        if (height[u] >= n){
            minCutSet.push_back(u);
        }
    } 
    return minCutSet;
    */
    cout<<"Iterations: "<<count<<endl;
    return excess[n-1];
}

int main(int argc, char const *argv[])
{   
    /*
    // Controllo se il numero di argomenti è corretto
    if (argc != 3) {
        printf("Utilizzo: %s <x> <N>\n", argv[0]);
        return 1;
    }*/
    
    // Dichiarazione delle variabili

    int n = 4;
    int s = 0;
    int t = n-1;
    int *capacity = (int *)malloc(n*n*sizeof(int));
    int *flow = (int *)malloc(n*n*sizeof(int));
    int *excess = (int *)malloc(n*sizeof(int));
    int *height = (int *)malloc(n*sizeof(int));

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

    /*vector<int> minCutSet = pushRelabel(capacity, flow, excess, height, n, s, t);
    cout<<"Min cut set: ";
    for (int i = 0; i < minCutSet.size(); i++){
        cout<<minCutSet[i]<<" ";
    }
    cout<<endl;*/

    int maxFlow = pushRelabel(capacity, flow, excess, height, n, s, t);
    cout<<"Max flow: "<<maxFlow<<endl;

    return 0;
}
