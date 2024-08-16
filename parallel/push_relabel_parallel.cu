#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <list>

using namespace std;

#define INF 1000000

void initialize(int *capacity, int *excess, int *height, int *residual, int *totalExcess, int n, int s){ //TODO: questo è uguale
    for (int i = 0; i < n; i++){
        height[i] = 0;
        excess[i] = 0;
    }
    
    height[s] = n;
    *totalExcess = 0;

    // Calcolo del preflusso
    for (int i = 0; i < n; i++){
        if (capacity[s*n + i] > 0){
            residual[s*n + i] = 0;
            residual[i*n + s] = capacity[s*n + i] + capacity[i*n +s]; // TODO: check aumentare la capacità
            excess[i] = capacity[s*n + i];
            *totalExcess += capacity[s*n + i];
        }
    }
}

__global__ void pushKernel(int *d_capacity, int *d_excess, int *d_height, int *d_residual, int n){  //TODO: cambiano solo le dichiarazioni delle variabili
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < n){
        int cycle = n; //KERNEL_CYCLES
        //TODO: vengono dichiarate qua le variabili ma non dovrebbe cambiare nulla
        int e1, h1, h2, v1, delta;
        while (cycle>0){
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
        cycle--;
        }
    }
}

void globalRelabel(int *capacity, int *excess, int *height, int *residual, int *totalExcess, bool *scanned, bool *mark, int n, int t){
    for (int u = 0; u < n; u++){
        for (int v = 0; v < n; v++){
            //TODO: qua c'è un if che controlla capacity[u,v] > 0
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
    int x, y, current; //TODO: togliere y e cambiare indice for
    
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
    cout<<"Scanned"<<endl;
    for (int i = 0; i < n; i++)
    {
        cout<<scanned[i]<<" ";
    }
    cout<<endl;

    cout<<"Mark"<<endl;
    for (int i = 0; i < n; i++)
    {
        cout<<mark[i]<<" ";
    }
    cout<<endl;
    
    bool allScanned = true;
    for(int i = 0; i < n; i++){
        if(scanned[i]==false){
            allScanned = false;
            break;
        }
    }
    if(allScanned==false){
        cout<<"AllScanned false"<<endl;
        for (int i = 0; i < n; i++){
            if(!(scanned[i] && mark[i])){ //TODO: qua viene usato !(scanned[u] OR mark[u]) che è uguale a !scanned[u] AND !mark[u]
                cout<<"Sono entrato qui"<<endl;
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

    int count = 0;
    
    while ((excess[s]+excess[t])<*totalExcess){
        cout<<"Eccesso S "<<excess[s]<<endl;
        cout<<"Eccesso T "<<excess[t]<<endl;
        cudaMemcpy(d_height, height, n*sizeof(int), cudaMemcpyHostToDevice);
        pushKernel<<<gridSize, blockSize>>>(d_capacity, d_excess, d_height, d_residual, n);
        cudaDeviceSynchronize();
        cudaMemcpy(excess, d_excess, n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(height, d_height, n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(residual, d_residual, n*n*sizeof(int), cudaMemcpyDeviceToHost);
        cout<<"Eccesso prima global relabel "<<*totalExcess<<endl;
        globalRelabel(capacity, excess, height, residual, totalExcess, scanned, mark, n, t);
        cout<<"Eccesso dopo global relabel "<<*totalExcess<<endl;
        cout<<endl;
        if(count == 4){
            //break;
        }
        count++;
    }
    return excess[t];
}

int main(int argc, char const *argv[]){ //TODO: fa i CUDA malloc e cudaMemcpy nel main e poi cudaFree alla fine
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
    residual[0] = 0;
    residual[1] = 2;
    residual[2] = 4;
    residual[3] = 0;
    residual[4] = 0;
    residual[5] = 0;
    residual[6] = 3;
    residual[7] = 1;
    residual[8] = 0;
    residual[9] = 0;
    residual[10] = 0;
    residual[11] = 5;
    residual[12] = 0;
    residual[13] = 0;
    residual[14] = 0;
    residual[15] = 0;


    int *d_capacity, *d_excess, *d_height, *d_residual;    

    cudaMalloc((void**)&d_capacity, n*n*sizeof(int));
    cudaMalloc((void**)&d_excess, n*sizeof(int)); //TODO: qua usa cudaMalloc((void**)&gpu_height,V*sizeof(int)); e inizializza dopo
    cudaMalloc((void**)&d_height, n*sizeof(int));
    cudaMalloc((void**)&d_residual, n*n*sizeof(int));

    initialize(capacity, excess, height, residual, totalExcess, n, s);
    
    std::cout<<"Eccessi"<<endl;
    for (int i = 0; i < n; i++)
    {
        std::cout<<excess[i]<<" ";
    }
    std::cout<<"Fine eccessi"<<endl;
    std::cout<<"Altezze"<<endl;
    for (int i = 0; i < n; i++)
    {
        std::cout<<height[i]<<" ";
    }
    std::cout<<"Fine altezze"<<endl;
    std::cout<<"Residui"<<endl;
    for (int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++){
            std::cout<<residual[i*n + j]<<" ";
        }
        std::cout<<endl;
    }
    std::cout<<"Fine residui"<<endl;
    std::cout<<"Total excess: "<<*totalExcess<<endl;
    
    
    cudaMemcpy(d_capacity, capacity, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_excess, excess, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_height, height, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, residual, n*n*sizeof(int), cudaMemcpyHostToDevice);
    
    int maxFlow = pushRelabel(capacity, excess, height, residual, d_capacity, d_excess, d_height, d_residual, totalExcess, n, s, t);
    std::cout<<"Max flow: "<<maxFlow<<endl;


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
