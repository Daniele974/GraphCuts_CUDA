#include "../include/push_relabel_parallel.cuh"

using namespace cooperative_groups;

void preflow(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess){

    // Inizializzazione di height e excess flow
    for(int i = 0; i < V; i++)
    {
        height[i] = 0; 
        excess[i] = 0;
    }
    
    height[source] = V;
    *totalExcess = 0;

    // Inizializzazione del preflow da source a tutti i nodi adiacenti
    for(int i = 0; i < V; i++)
    {
        if(capacities[source*V + i] > 0)
        {
            residual[source*V + i] = 0;
            residual[i*V + source] = capacities[source*V + i] + capacities[i*V + source];
            
            excess[i] = capacities[source*V + i];
            *totalExcess += excess[i];
        } 
    }

}

__global__ void pushRelabelKernel(int V, int source, int sink, int *d_capacities, int *d_residual, int *d_height,int *d_excess){
    grid_group grid = this_grid();
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

    int cycle = V;

    while (cycle > 0) {

        for (int u = idx; u < V; u += blockDim.x * gridDim.x) {

            int e1, h1, h2, delta;
            int v,v1;

            if( (d_excess[u] > 0) && (d_height[u] < V) && u != source && u != sink){
                e1 = d_excess[u];
                h1 = INF;
                v1 = -1;

                for(v = 0; v < V; v++){

                    if(d_residual[u*V + v] > 0){
                        h2 = d_height[v];
                        if(h2 < h1){
                            v1 = v;
                            h1 = h2;
                        }
                    }
                }
                if (v1 == -1) {
                    d_height[u] = V;
                } else {
                    if(d_height[u] > h1){
                        
                        //TODO: usare min
                        delta = e1;
                        if(e1 > d_residual[u*V + v1]){
                            delta = d_residual[u*V + v1];
                        }

                        atomicAdd(&d_residual[v1*V + u],delta);
                        atomicSub(&d_residual[u*V + v1],delta);

                        atomicAdd(&d_excess[v1],delta);
                        atomicSub(&d_excess[u],delta);
                    } else {
                        d_height[u] = h1 + 1;
                    }
                }
            }
        }

        cycle = cycle - 1;
        grid.sync();
    }
}

void globalRelabel(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess, bool *mark, bool *scanned){
    for(int u = 0; u < V; u++){
        for(int v = 0; v < V; v++){
            if(capacities[u*V + v] > 0)
            {
                if(height[u] > height[v] + 1)
                {
                    excess[u] = excess[u] - residual[u*V + v];
                    excess[v] = excess[v] + residual[u*V + v];
                    residual[v*V + u] = residual[v*V + u] + residual[u*V + v];
                    residual[u*V + v] = 0;
                }
            }
        }
    }

    // BFS
    std::list<int> Queue;
    int x,y,current;
        
    for(int i = 0; i < V; i++)
    {
        scanned[i] = false;
    }

    // Accodo sink
    Queue.push_back(sink);
    scanned[sink] = true;
    height[sink] = 0;

    while(!Queue.empty())
    {
        x = Queue.front();
        Queue.pop_front();

        current = height[x];
        current = current + 1;

        for(y = 0; y < V; y++){
            if(residual[y*V+x] > 0){
                if(scanned[y] == false){
                    height[y] = current;
                    scanned[y] = true;
                    Queue.push_back(y);
                }
            }
        }
    }

    // Controllo se tutti i nodi sono stati scansionati
    bool allScanned = true;
    for(int i = 0; i < V; i++){
        if(scanned[i] == false){
            allScanned = false;
            break;
        }
    }

    // Se non tutti i nodi sono stati scansionati...
    if(allScanned == false){
        for(int i = 0; i < V; i++){

            if(!(scanned[i] || mark[i])){
                mark[i] = true;
                *totalExcess = *totalExcess - excess[i];
            }
        }
    }
}

void pushRelabel(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess, int *d_capacities, int *d_residual, int *d_height, int *d_excess){
    
    // Dichiarazione delle variabili per global relabel
    bool *mark,*scanned;
    mark = (bool*)malloc(V*sizeof(bool));
    scanned = (bool*)malloc(V*sizeof(bool));

    // Configure the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    dim3 num_blocks(deviceProp.multiProcessorCount); // un blocco per ogni SM
    dim3 block_size(256); // 256 threads per blocco

    void* kernel_args[] = {&V,&source,&sink,&d_capacities,&d_residual,&d_height,&d_excess};

    // initialising mark values to false for all nodes
    for(int i = 0; i < V; i++)
    {
        mark[i] = false;
    }

    while((excess[source] + excess[sink]) < *totalExcess)
    {
        // Trasferimento dati da host a device
        HANDLE_ERROR(cudaMemcpy(d_height,height,V*sizeof(int),cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_excess, excess,V*sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_residual,residual,V*V*sizeof(int),cudaMemcpyHostToDevice));

        cudaError_t cudaStatus;
        cudaStatus = cudaLaunchCooperativeKernel((void*)pushRelabelKernel, num_blocks, block_size, kernel_args, 0, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error, for example, by cleaning up resources and exiting
            exit(1);
        }

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s.\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Trasferimento dati da device a host
        HANDLE_ERROR(cudaMemcpy(height,d_height,V*sizeof(int),cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(excess,d_excess,V*sizeof(int),cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(residual,d_residual,V*V*sizeof(int),cudaMemcpyDeviceToHost));
        
        // Global relabel
        globalRelabel(V,source,sink,capacities,residual,height,excess,totalExcess,mark,scanned);
    }
}

std::vector<int> findMinCutSetFromT(int n, int t, int *residual){
    std::vector<int> minCutSet;
    std::queue<int> q;
    std::vector<bool> visited(n, false);
    minCutSet.push_back(t);
    q.push(t);
    visited[t] = true;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v = 0; v < n; ++v) {
            if (!visited[v] && residual[v*n + u] > 0) {
                minCutSet.push_back(v);
                q.push(v);
                visited[v] = true;
            }
        }
    }

    return minCutSet;
}

int executePushRelabel(std::string filename, std::string output){
    //Dichiarazione degli eventi per la misurazione del tempo
    cudaEvent_t startEvent, endInitializationEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endInitializationEvent);
    cudaEventCreate(&endEvent);


    // Dichiarazione delle variabili host
    int V, E, source, sink;
    int *capacities, *residual, *height, *excess;
    int *totalExcess;

    // Dichiarazione delle variabili device
    int *d_capacities, *d_residual, *d_height, *d_excess;

    // Lettura del grafo da file
    readGraphFromFile(filename, V, &capacities);

    cudaEventRecord(startEvent, 0);
    
    // Inizializzazione delle variabili source e sink
    source = 0;
    sink = V - 1;

    // Inizializzazione delle variabili E e residual
    residual = (int*)malloc(V*V*sizeof(int));

    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++){
            residual[i*V + j] = capacities[i*V + j];
            if(capacities[i*V + j] > 0){
                E = E + 1;
            }
        }
    }

    // Inizializzazione delle variabili height e excess
    height = (int*)malloc(V*sizeof(int));
    excess = (int*)malloc(V*sizeof(int));
    totalExcess = (int*)malloc(sizeof(int));

    // Allocazione delle variabili device
    HANDLE_ERROR(cudaMalloc((void**)&d_capacities,V*V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_residual,V*V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_height,V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_excess,V*sizeof(int)));

    // Calcolo preflow
    preflow(V,source,sink,capacities,residual,height,excess,totalExcess);

    // Trasferimento dati da host a device
    HANDLE_ERROR(cudaMemcpy(d_height,height,V*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_excess,excess,V*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_capacities,capacities,V*V*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_residual,residual,V*V*sizeof(int),cudaMemcpyHostToDevice));

    cudaEventRecord(endInitializationEvent, 0);
    pushRelabel(V,source,sink,capacities,residual,height,excess,totalExcess,d_capacities,d_residual,d_height,d_excess);
    cudaEventRecord(endEvent, 0);

    // Misurazione del tempo
    cudaEventSynchronize(endEvent);
    float initializationTime = 0.0f;
    float executionTime = 0.0f;
    float totalTime = 0.0f;
    cudaEventElapsedTime(&initializationTime, startEvent, endInitializationEvent);
    cudaEventElapsedTime(&executionTime, endInitializationEvent, endEvent);
    cudaEventElapsedTime(&totalTime, startEvent, endEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endInitializationEvent);
    cudaEventDestroy(endEvent);
    
    // Calcolo del min cut set
    std::vector<int> minCut = findMinCutSetFromT(V, sink, residual);

    // Scrittura dei risultati su file
    writeResultsToFile(output, excess[sink], minCut, initializationTime, executionTime, totalTime);
    
    // Liberazione della memoria
    cudaFree(d_capacities);
    cudaFree(d_residual);
    cudaFree(d_height);
    cudaFree(d_excess);
    
    free(capacities);
    free(residual);
    free(height);
    free(excess);
    free(totalExcess);

    return 0;
}