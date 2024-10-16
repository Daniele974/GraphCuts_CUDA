#include "../include/push_relabel_parallel.cuh"

using namespace cooperative_groups;

void preflow(int V, int source, int sink, int *offset, int *column, int *capacities, int *residual, int *height, int *excess, int *totalExcess){

    // Inizializzazione di height e excess flow
    for(int i = 0; i < V; i++)
    {
        height[i] = 0; 
        excess[i] = 0;
    }
    
    height[source] = V;
    *totalExcess = 0;

    // Inizializzazione del preflow da source a tutti i nodi adiacenti (push)
    for(int i = offset[source]; i < offset[source+1]; i++){
        int neighbor = column[i];
        if(capacities[i] > 0){
            residual[i] = 0;

            for(int j = offset[neighbor]; j < offset[neighbor+1]; j++){
                if(column[j] == source){
                    residual[j] = capacities[i];
                    break;
                }
            }

            excess[neighbor] = capacities[i];
            *totalExcess = *totalExcess + excess[neighbor];
        }
    }
}

__global__ void pushRelabelKernel(int V, int source, int sink, int *d_offset, int *d_column, int *d_capacities, int *d_residual, int *d_height,int *d_excess){
    // Cooperative groups
    grid_group grid = this_grid();
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

    int cycle = V;

    while (cycle > 0) {
        
        for (int u = idx; u < V; u += blockDim.x * gridDim.x) {

            int e1, h1, h2, delta;
            int v,v1,v1idx;

            // Se u è un nodo interno e ha un eccesso di flusso (nodo attivo)
            if((d_excess[u] > 0) && (d_height[u] < V) && u != source && u != sink){
                e1 = d_excess[u];
                h1 = INF;
                v1 = -1;
                v1idx = -1;

                // Ricerca nodo adiacente con altezza minore
                for(int i = d_offset[u]; i < d_offset[u+1]; i++){
                    v = d_column[i];
                    if(d_residual[i] > 0){
                        h2 = d_height[v];
                        if(h2 < h1){
                            v1 = v;
                            h1 = h2;
                            v1idx = i;
                        }
                    }
                    
                }

                // Se non è stato trovato un nodo adiacente con altezza minore
                if (v1 == -1) {
                    d_height[u] = V;
                } else { 

                    if(d_height[u] > h1){

                        int backwardIdx = -1;

                        // Ricerca arco di ritorno usando la ricerca binaria
                        int left = d_offset[v1];
                        int right = d_offset[v1 + 1] - 1;
                        
                        while (left <= right) {
                            int mid = left + (right - left) / 2;
                            if (d_column[mid] == u) {
                                backwardIdx = mid;  // Arco di ritorno trovato
                                break;
                            } else if (d_column[mid] < u) {
                                left = mid + 1;     // Cerca nella parte destra
                            } else {
                                right = mid - 1;    // Cerca nella parte sinistra
                            }
                        }

                        // Se l'arco di ritorno non è stato trovato, errore
                        if(backwardIdx == -1){
                            printf("Error: backward edge not found\n");
                            return;
                        }

                        // Push
                        // Calcolo del delta (quantità di flusso da spostare)
                        delta = e1;
                        if(e1 > d_residual[v1idx]){
                            delta = d_residual[v1idx];
                        }

                        // Aggiornamento dei valori di residual ed excess
                        atomicAdd(&d_residual[backwardIdx],delta);
                        atomicSub(&d_residual[v1idx],delta);

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

void globalRelabel(int V, int source, int sink, int *offset, int *column, int *capacities, int *residual, int *height, int *excess, int *totalExcess, bool *mark, bool *scanned){
    
    for(int u = 0; u < V; u++){
        for(int i = offset[u]; i < offset[u+1]; i++){
            int v = column[i];
            if(height[u] > height[v]+1){
                int flow;
                if(excess[u] < residual[i]){
                    flow = excess[u];
                }else{
                    flow = residual[i];
                }
                excess[u] -= flow;
                excess[v] += flow;
                residual[i] -= flow;
            }
        }
    }

    // BFS da sink per ricalcolare le altezze
    std::list<int> Queue;
    int x,y,current;
    
    // Inizializzazione della variabile scanned
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
            for(int i = offset[y]; i < offset[y+1]; i++){
                if(column[i] == x && residual[i] > 0){
                    if(scanned[y] == false){
                        height[y] = current;
                        scanned[y] = true;
                        Queue.push_back(y);
                    }
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
        // Marca i nodi non scansionati e aggiorna il valore di totalExcess
        for(int i = 0; i < V; i++){

            if(!(scanned[i] || mark[i])){
                mark[i] = true;
                *totalExcess = *totalExcess - excess[i];
            }
        }
    }
}

void pushRelabel(int V, int E, int source, int sink, int *offset, int *column, int *capacities, int *residual, int *height, int *excess, int *totalExcess, int *d_offset, int *d_column, int *d_capacities, int *d_residual, int *d_height, int *d_excess){
    
    // Dichiarazione delle variabili per global relabel
    bool *mark,*scanned;
    mark = (bool*)malloc(V*sizeof(bool));
    scanned = (bool*)malloc(V*sizeof(bool));

    // Configurazione della GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    dim3 num_blocks(deviceProp.multiProcessorCount); // un blocco per ogni SM
    dim3 block_size(256); // 256 threads per blocco

    void* kernel_args[] = {&V,&source,&sink,&d_offset,&d_column,&d_capacities,&d_residual,&d_height,&d_excess};

    // Inizializzazione dei valori di mark
    for(int i = 0; i < V; i++)
    {
        mark[i] = false;
    }

    // Esecuzione dell'algoritmo push-relabel
    while((excess[source] + excess[sink]) < *totalExcess)
    {
        // Trasferimento dati da host a device
        HANDLE_ERROR(cudaMemcpy(d_height,height,V*sizeof(int),cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_excess, excess,V*sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_residual,residual,E*sizeof(int),cudaMemcpyHostToDevice));

        // Esecuzione del kernel
        cudaError_t cudaStatus;
        cudaStatus = cudaLaunchCooperativeKernel((void*)pushRelabelKernel, num_blocks, block_size, kernel_args, 0, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }

        cudaDeviceSynchronize();

        // Controllo errori
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s.\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Trasferimento dati da device a host
        HANDLE_ERROR(cudaMemcpy(height,d_height,V*sizeof(int),cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(excess,d_excess,V*sizeof(int),cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(residual,d_residual,E*sizeof(int),cudaMemcpyDeviceToHost));
        
        // Global relabel
        globalRelabel(V,source,sink,offset,column,capacities,residual,height,excess,totalExcess,mark,scanned);
    }
}


void computeTranspose(int V, int E, int *offset, int *column, int *forwardFlow, int *t_offset, int *t_column, int *t_forwardFlow) {

    for(int i = 0; i < E; i++) {
        t_offset[column[i]+2]++;
    }
    
    for(int i = 2; i < V+1; i++) {
        t_offset[i] += t_offset[i-1];
    }
    
    for(int i = 0; i < V; i++) {
        for(int j = offset[i]; j < offset[i+1]; j++) {
            int v = column[j];
            int newIndex = t_offset[v+1]++;
            t_column[newIndex] = i;
            t_forwardFlow[newIndex] = forwardFlow[j];
        }
    }
}

std::vector<int> findMinCutSetFromSinkOMP(int V, int E, int sink, int *offset, int *column, int *forwardFlow){

    int *t_offset = (int*)malloc((V+1)*sizeof(int));
    int *t_column = (int*)malloc(E*sizeof(int));
    int *t_forwardFlow = (int*)malloc(E*sizeof(int));

    for(int i = 0; i < V+1; i++){
        t_offset[i] = 0;
    }

    for(int i = 0; i < E; i++){
        t_column[i]= 0;
        t_forwardFlow[i] = 0;
    }

    computeTranspose(V, E, offset, column, forwardFlow, t_offset, t_column, t_forwardFlow);

    std::vector<int> minCutSet;
    std::vector<int> vertexList;
    bool *visited = (bool*)malloc(V*sizeof(bool));

    for(int i = 0; i < V; i++) {
        visited[i] = false;
    }

    minCutSet.push_back(sink);
    vertexList.push_back(sink);
    visited[sink] = true;

    while (!vertexList.empty()) {
        std::vector<int> newVertexList;
        
        #pragma omp parallel
        {
            std::vector<int> localVertexList;
            std::vector<int> localMinCutSet;
            
            #pragma omp for nowait schedule(dynamic)
            for (int i = 0; i < vertexList.size(); i++) {
                int u = vertexList[i];
                
                for (int j = t_offset[u]; j < t_offset[u+1]; j++) {
                    int v = t_column[j];
                    bool shouldAdd = false;

                    #pragma omp critical (checkVisited)
                    {
                        if(!visited[v] && t_forwardFlow[j] > 0) {
                            visited[v] = true;
                            shouldAdd = true;
                        }
                    }
                    
                    if(shouldAdd) {
                        localMinCutSet.push_back(v);
                        localVertexList.push_back(v);
                    }
                }
            }

            #pragma omp critical (insert)
            {
                minCutSet.insert(minCutSet.end(), localMinCutSet.begin(), localMinCutSet.end());
                newVertexList.insert(newVertexList.end(), localVertexList.begin(), localVertexList.end());
            }
        }
        vertexList = newVertexList;
    }

    return minCutSet;
}

int executePushRelabel(std::string filename, std::string output, bool computeMinCut){
    //Dichiarazione degli eventi per la misurazione del tempo
    cudaEvent_t startEvent, endInitializationEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endInitializationEvent);
    cudaEventCreate(&endEvent);


    // Dichiarazione delle variabili host
    int V, E, realE, source, sink;
    int *offset, *column, *capacities, *residual;
    int *height, *excess;
    int *totalExcess;

    // Dichiarazione delle variabili device
    int *d_offset, *d_column, *d_capacities, *d_residual, *d_height, *d_excess;

    // Lettura del grafo da file
    // Controllo estensione file e lettura grafo
    if(filename.find(".txt") != std::string::npos){
        readBCSRGraphFromFile(filename, V, E, realE, source, sink, &offset, &column, &capacities, &residual);
    }else if(filename.find(".max") != std::string::npos){
        readBCSRGraphFromDIMACSFile(filename, V, E, realE, source, sink, &offset, &column, &capacities, &residual);
    }else{
        std::cerr << "Error: file format not supported" << std::endl;
        return 1;
    }

    cudaEventRecord(startEvent, 0);

    // Inizializzazione delle variabili height, excess e totalExcess
    height = (int*)malloc(V*sizeof(int));
    excess = (int*)malloc(V*sizeof(int));
    totalExcess = (int*)malloc(sizeof(int));

    // Allocazione delle variabili device
    HANDLE_ERROR(cudaMalloc((void**)&d_offset, (V+1)*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_column, E*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_capacities,E*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_residual,E*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_height,V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_excess,V*sizeof(int)));

    // Calcolo preflow
    preflow(V,source,sink,offset,column,capacities,residual,height,excess,totalExcess);

    // Trasferimento dati da host a device
    HANDLE_ERROR(cudaMemcpy(d_offset,offset,(V+1)*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_column,column,E*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_capacities,capacities,E*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_residual,residual,E*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_height,height,V*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_excess,excess,V*sizeof(int),cudaMemcpyHostToDevice));


    cudaEventRecord(endInitializationEvent, 0);
    // Esecuzione dell'algoritmo push-relabel
    pushRelabel(V,E,source,sink,offset,column,capacities,residual,height,excess,totalExcess,d_offset,d_column,d_capacities,d_residual,d_height,d_excess);
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
    std::vector<int> minCut = {};
    if(computeMinCut){
        minCut = findMinCutSetFromSinkOMP(V, E, sink, offset, column, residual);
    }

    // Scrittura dei risultati su file
    writeResultsToFile(output, excess[sink], minCut, initializationTime, executionTime, totalTime, V, realE);
    
    // Liberazione della memoria
    cudaFree(d_offset);
    cudaFree(d_column);
    cudaFree(d_capacities);
    cudaFree(d_residual);
    cudaFree(d_height);
    cudaFree(d_excess);
    
    free(offset);
    free(column);
    free(capacities);
    free(residual);
    free(height);
    free(excess);
    free(totalExcess);

    return 0;
}