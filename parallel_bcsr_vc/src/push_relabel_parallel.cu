#include "../include/push_relabel_parallel.cuh"

// Variabile globale per dimensione coda avq
__device__ unsigned int avqSize;

void preflow(int V, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, int *totalExcess){
    
    // Inizializzazione altezze e eccedenze dei nodi
    for (int i = 0; i < V; i++){
        height[i] = 0;
        excess[i] = 0;
    }

    // Inizializzazione altezza del nodo sorgente
    height[source] = V;

    // Inizializzazione totalExcess
    *totalExcess = 0;

    // Inizializzazione flusso iniziale (da sorgente a vicini)
    for(int i = offset[source]; i < offset[source+1]; i++){
        int neighbor = column[i];
        if(capacities[i] > 0){
            forwardFlow[i] = 0;
            
            for(int j = offset[neighbor]; j < offset[neighbor+1]; j++){
                if(column[j] == source){
                    forwardFlow[j] = capacities[i];
                    break;
                }
            }

            excess[neighbor] = capacities[i];
            *totalExcess = *totalExcess + excess[neighbor];
        }
    }
}

__device__ void scanActiveVertices(int V, int source, int sink, int *d_height, int *d_excess, int *d_avq){
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    grid_group grid = this_grid();
    
    if(idx == 0){
        avqSize = 0;
    }
    grid.sync();
    
    // Scansione dei nodi attivi
    for(int u = idx; u < V; u += blockDim.x * gridDim.x){

        // Controllo se u è attivo e interno
        if(d_excess[u] > 0 && d_height[u] < V && u != source && u != sink){
            int pos = atomicAdd(&avqSize, 1);
            d_avq[pos] = u;
        }
    }
}

template <unsigned int tileSize> __device__  int tiledSearchNeighbor(thread_block_tile <tileSize> tile, int pos, int *s_height, int *s_vid, int *s_vidx, int *Vindex, int V, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq){
    unsigned int idx = tile.thread_rank(); // indice del thread all'interno della tile

    int u = d_avq[pos];
    int degree = d_offset[u+1] - d_offset[u];
    int numIters = (int)ceilf((float)degree / (float)tileSize); // numero di iterazioni necessarie per scansionare tutti i vicini di u

    // Inizializzazione variabili per ricerca vicino con altezza minima
    int minH = INF;
    int minV = -1;

    // Inizializzazione shared memory
    s_height[threadIdx.x] = INF;
    s_vid[threadIdx.x] = -1;
    s_vidx[threadIdx.x] = -2;
    tile.sync();

    // Ricerca vicino con altezza minima
    for(int i = 0; i < numIters; i++){
        int vPos, v;
        if(i*tileSize + idx < degree){

            // Calcolo posizione del vicino
            vPos = d_offset[u] + i*tileSize + idx;
            v = d_column[vPos];

            // Controllo presenza di flusso tra u e v
            if(d_flows[vPos] > 0 && v != source){
                s_height[threadIdx.x] = d_height[v];
                s_vid[threadIdx.x] = v;
                s_vidx[threadIdx.x] = vPos;
            }else{
                s_height[threadIdx.x] = INF;
                s_vid[threadIdx.x] = -1;
                s_vidx[threadIdx.x] = -1;
            }
        }else{
            s_height[threadIdx.x] = INF;
            s_vid[threadIdx.x] = -1;
            s_vidx[threadIdx.x] = -1;
        }
        tile.sync();

        // Reduction per trovare altezza minima (e nodo associato)
        for(int s = tile.size() / 2; s > 0; s >>= 1){
            if(idx < s){ 
                if(s_height[threadIdx.x + s] < s_height[threadIdx.x]){
                    s_height[threadIdx.x] = s_height[threadIdx.x + s];
                    s_vid[threadIdx.x] = s_vid[threadIdx.x + s];
                    s_vidx[threadIdx.x] = s_vidx[threadIdx.x + s];
                }
            }
            tile.sync();
        }
        tile.sync();

        // Aggiornamento altezza minima e nodo associato
        if(idx == 0){
            if(minH > s_height[threadIdx.x]){
                minH = s_height[threadIdx.x];
                minV = s_vid[threadIdx.x];
                *Vindex = s_vidx[threadIdx.x];
            }
        }
        tile.sync();

        // Reset shared memory per prossima iterazione
        s_height[threadIdx.x] = INF;
        s_vid[threadIdx.x] = -1;
        tile.sync();
    }
    tile.sync();

    // Restituzione nodo con altezza minima
    return minV;
}

__global__ void pushKernel(int V, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq){
    grid_group grid = this_grid();
    thread_block block = this_thread_block();
    const int tileSize = 32;
    thread_block_tile<tileSize> tile = tiled_partition<tileSize>(block);
    int numTilesPerBlock = (blockDim.x + tileSize - 1) / tileSize;
    int numTilesPerGrid = numTilesPerBlock * gridDim.x;
    int tileIdx = blockIdx.x * numTilesPerBlock + block.thread_rank() / tileSize;

    // Inizializzazione variabili per vicino con altezza minima
    int minV = -1; // id nodo con altezza minima
    int Vindex = -1; // indice del nodo con altezza minima

    int cycle = V;

    // Shared memory
    extern __shared__ int shared[];
    int *s_height = shared;
    int *s_vid = (int *)&shared[blockDim.x];
    int *s_vidx = (int *)&s_vid[blockDim.x];

    while(cycle > 0){

        // Scansione dei nodi attivi
        scanActiveVertices(V, source, sink, d_height, d_excess, d_avq);
        grid.sync();

        if(avqSize == 0){
            break;
        }
        
        Vindex = -1;
        grid.sync();

        for(int i = tileIdx; i < avqSize; i += numTilesPerGrid){
            int u = d_avq[i];

            // Ricerca vicino con altezza minima
            minV = tiledSearchNeighbor<tileSize>(tile, i, s_height, s_vid, s_vidx, &Vindex, V, source, sink, d_height, d_excess, d_offset, d_column, d_capacities, d_flows, d_avq);
            tile.sync();
            
            if(tile.thread_rank() == 0){
                if(minV == -1){
                    d_height[u] = V;
                } else {
                    if(d_height[u] > d_height[minV]){
                        int d;
                        int backwardIdx = -1;

                        // Ricerca arco di ritorno usando la ricerca binaria
                        int left = d_offset[minV];
                        int right = d_offset[minV + 1] - 1;
                        
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

                        // Calcolo flusso da spostare (delta)
                        if(d_excess[u] > d_flows[Vindex]){
                            d = d_flows[Vindex];
                        } else {
                            d = d_excess[u];
                        }
                        
                        //Push
                        atomicAdd(&d_flows[backwardIdx], d);
                        atomicSub(&d_flows[Vindex], d);
                        atomicAdd(&d_excess[minV], d);
                        atomicSub(&d_excess[u], d);
                    } else {
                        d_height[u] = d_height[minV] + 1;
                    }
                }
            }
            tile.sync();
        }
        grid.sync();
        cycle--;
    }
}

__global__ void globalRelabelKernel(int V, int E, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_status, int *d_queue, int *d_queueSize, int *d_level, int *d_totalExcess, bool *terminate){
    grid_group grid = this_grid();
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Inizializzazione coda e altezza per global relabel
    if(idx == 0){
        d_status[sink] = 0;
        *d_queueSize = 0;
        *d_level = 1;
    }
    grid.sync();

    while(true){

        // Inserimento in coda dei nodi nella frontiera
        for (int i = idx; i < V; i += blockDim.x * gridDim.x) {
            if (d_status[i] == -1) {
                d_queue[atomicAdd(d_queueSize, 1)] = i;
            }
        }
        grid.sync();

        // Elaborazione dei nodi nella coda
        for(int i = idx; i < *d_queueSize; i+= blockDim.x * gridDim.x){
            int u = d_queue[i];

            // Ricerca vicini di u
            for(int j = d_offset[u]; j < d_offset[u+1]; j++){
                int v = d_column[j];

                // Aggiornamento altezza e status di u
                if(d_status[v] >= 0 && d_flows[j] > 0){
                    d_status[u] = *d_level + 1;
                    d_height[u] = d_status[u];
                    *terminate = false;
                    break;
                }
            }
        }
        grid.sync();

        // Controllo terminazione
        if(*terminate){
            break;
        }
        grid.sync();

        // Reset coda per prossima iterazione
        if(idx == 0){
            *d_queueSize = 0;
            *d_level = *d_level + 1;
            *terminate = true;
        }
        grid.sync();
    }
    grid.sync();

    // Aggiornamento di totalExcess
    for(int i = idx; i < V; i += blockDim.x * gridDim.x){
        if(d_status[i] == -1 && d_excess[i] > 0 && i != source && i != sink){
            atomicSub(d_totalExcess, d_excess[i]);
            d_excess[i] = 0;
        }
    }
}

void globalRelabel(int V, int E, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, 
                    int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *totalExcess, bool *mark, bool *scanned){
    
    for(int u = 0; u < V; u++){
        for(int i = offset[u]; i < offset[u+1]; i++){
            int v = column[i];
            if(height[u] > height[v]+1){
                int flow;
                if(excess[u] < forwardFlow[i]){
                    flow = excess[u];
                }else{
                    flow = forwardFlow[i];
                }
                excess[u] -= flow;
                excess[v] += flow;
                forwardFlow[i] -= flow;
            }
        }
    }
    
    // Strutture per global relabel
    int *d_status, *d_queue, *d_level, *d_totalExcess, *d_queueSize;
    bool *terminate;

    // Allocazione memoria
    HANDLE_ERROR(cudaMalloc((void**)&d_status, V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_queue, V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_level, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_totalExcess, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_queueSize, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&terminate, sizeof(bool)));

    // Inizializzazione strutture
    HANDLE_ERROR(cudaMemset(d_status, -1, V*sizeof(int)));
    HANDLE_ERROR(cudaMemset(d_queue, 0, V*sizeof(int)));
    HANDLE_ERROR(cudaMemset(d_level, 0, sizeof(int)));
    HANDLE_ERROR(cudaMemset(d_queueSize, 0, sizeof(int)));
    HANDLE_ERROR(cudaMemset(terminate, true, sizeof(bool)));

    // Trasferimento dati da host a device
    HANDLE_ERROR(cudaMemcpy(d_height, height, V*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_excess, excess, V*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_flows, forwardFlow, E*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_totalExcess, totalExcess, sizeof(int), cudaMemcpyHostToDevice));

    // Configurazione GPU
    int device = -1;
    HANDLE_ERROR(cudaGetDevice(&device));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    dim3 num_blocks(prop.multiProcessorCount * 1);  // 1 blocco per SM
    dim3 block_size(1024);    

    // Argomenti kernel global relabel
    void *kernel_args[] = {&V, &E, &source, &sink, &d_height, &d_excess, &d_offset, &d_column, &d_capacities, &d_flows, &d_status, &d_queue, &d_queueSize, &d_level, &d_totalExcess, &terminate};

    // Lancio kernel global relabel
    cudaError_t cudaStatus;
    cudaStatus = cudaLaunchCooperativeKernel((void*)globalRelabelKernel, num_blocks, block_size, kernel_args, 0, 0);

    // Controllo errori
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    cudaDeviceSynchronize();

    // Trasferimento dati da device a host
    HANDLE_ERROR(cudaMemcpy(height, d_height, V*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(excess, d_excess, V*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(forwardFlow, d_flows, E*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(totalExcess, d_totalExcess, sizeof(int), cudaMemcpyDeviceToHost));

    // Liberazione memoria
    HANDLE_ERROR(cudaFree(d_status));
    HANDLE_ERROR(cudaFree(d_queue));
    HANDLE_ERROR(cudaFree(d_level));
    HANDLE_ERROR(cudaFree(d_totalExcess));
    HANDLE_ERROR(cudaFree(d_queueSize));
    HANDLE_ERROR(cudaFree(terminate));
}

int pushRelabel(int V, int E, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, int *totalExcess, 
                int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq){
    
    // Allocazione memoria per strutture per global relabel
    bool *mark, *scanned;
    mark = (bool *)malloc(V*sizeof(bool));
    scanned = (bool *)malloc(V*sizeof(bool));

    // Inizializzazione variabile mark
    for(int i = 0; i < V; i++){
        mark[i] = false;
    }

    // Configurazione GPU
    int device = -1;
    HANDLE_ERROR(cudaGetDevice(&device));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    dim3 num_blocks(prop.multiProcessorCount * 1);  // 1 blocco per SM
    dim3 block_size(1024);  

    // Configurazione shared memory per kernel push
    size_t sharedMemSize = 3 * block_size.x * sizeof(int);

    // Configurazione argomenti kernel push
    void* kernel_args[] = {&V, &source, &sink, &d_height, &d_excess, 
                        &d_offset, &d_column, &d_capacities, &d_flows, &d_avq};
    
    // Esecuzione algoritmo push-relabel
    while(excess[source] + excess[sink] < *totalExcess){

        // Trasferimento dati da host a device
        HANDLE_ERROR(cudaMemcpy(d_height, height, V*sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_excess, excess, V*sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_flows, forwardFlow, E*sizeof(int), cudaMemcpyHostToDevice));
        
        // Lancio kernel push
        cudaError_t cudaStatus;
        cudaStatus = cudaLaunchCooperativeKernel((void*)pushKernel, num_blocks, block_size, kernel_args, sharedMemSize, 0);
        
        // Controllo errori
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }
        cudaDeviceSynchronize();

        // Trasferimento dati da device a host
        HANDLE_ERROR(cudaMemcpy(height, d_height, V*sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(excess, d_excess, V*sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(forwardFlow, d_flows, E*sizeof(int), cudaMemcpyDeviceToHost));

        // Global relabel
        globalRelabel(V, E, source, sink, height, excess, offset, column, capacities, forwardFlow, d_height, d_excess, d_offset, d_column, d_capacities, d_flows, totalExcess, mark, scanned); 
    }

    // Return del flusso massimo
    return excess[sink];
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
    
    // Dichiarazione variabili host
    int V, E, realE, source, sink;
    int *offset, *column, *capacities, *forwardFlow;
    int *height, *excess,  *totalExcess, *avq;

    // Controllo estensione file e lettura grafo
    if(filename.find(".txt") != std::string::npos){
        readBCSRGraphFromFile(filename, V, E, realE, source, sink, &offset, &column, &capacities, &forwardFlow);
    }else if(filename.find(".max") != std::string::npos){
        readBCSRGraphFromDIMACSFile(filename, V, E, realE, source, sink, &offset, &column, &capacities, &forwardFlow);
    }else{
        std::cerr << "Error: file format not supported" << std::endl;
        return 1;
    }

    // Primo evento per la misurazione del tempo
    cudaEventRecord(startEvent, 0);

    // Allocazione memoria host
    height = (int *)malloc(V*sizeof(int));
    excess = (int *)malloc(V*sizeof(int));
    totalExcess = (int *)malloc(sizeof(int));
    avq = (int *)malloc(V*sizeof(int));
    
    // Inizializzazione coda avq (coda vertici attivi)
    for (int i = 0; i < V; i++)
    {
        avq[i] = 0;
    }

    // Dichiarazione variabili device
    int *d_height, *d_excess, *d_column, *d_offset, *d_capacities, *d_flows, *d_avq;

    // Allocazione memoria device
    HANDLE_ERROR(cudaMalloc((void**)&d_height, V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_excess, V*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_column, E*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_offset, (V+1)*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_capacities, E*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_flows, E*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_avq, V*sizeof(int)));
   
    // Preflow
    preflow(V, source, sink, height, excess, offset, column, capacities, forwardFlow, totalExcess);

    // Trasferimento dati da host a device
    HANDLE_ERROR(cudaMemcpy(d_height, height, V*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_excess, excess, V*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_offset, offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_column, column, E*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_capacities, capacities, E*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_flows, forwardFlow, E*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_avq, avq, V*sizeof(int), cudaMemcpyHostToDevice));

    // Secondo evento per la misurazione del tempo
    cudaEventRecord(endInitializationEvent, 0);

    // Algoritmo push-relabel
    int maxFlow = pushRelabel(V, E, source, sink, height, excess, offset, column, capacities, forwardFlow, totalExcess, d_height, d_excess, d_offset, d_column, d_capacities, d_flows, d_avq);
    
    // Terzo evento per la misurazione del tempo
    cudaEventRecord(endEvent, 0);
    
    // Attendo la fine dell'evento endEvent
    cudaEventSynchronize(endEvent);
    
    // Misurazione del tempo
    float initializationTime = 0.0f;
    float executionTime = 0.0f;
    float totalTime = 0.0f;
    cudaEventElapsedTime(&initializationTime, startEvent, endInitializationEvent);
    cudaEventElapsedTime(&executionTime, endInitializationEvent, endEvent);
    cudaEventElapsedTime(&totalTime, startEvent, endEvent);

    // Distruzione degli eventi
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endInitializationEvent);
    cudaEventDestroy(endEvent);

    // Calcolo del taglio minimo
    std::vector<int> minCut = {};
    
    if(computeMinCut){
        minCut = findMinCutSetFromSinkOMP(V, E, sink, offset, column, forwardFlow);
    }

    // Scrittura risultati su file
    writeResultsToFile(output, maxFlow, minCut, initializationTime, executionTime, totalTime, V, realE);

    // Liberazione memoria device
    HANDLE_ERROR(cudaFree(d_height));
    HANDLE_ERROR(cudaFree(d_excess));
    HANDLE_ERROR(cudaFree(d_offset));
    HANDLE_ERROR(cudaFree(d_column));
    HANDLE_ERROR(cudaFree(d_capacities));
    HANDLE_ERROR(cudaFree(d_flows));
    HANDLE_ERROR(cudaFree(d_avq));

    // Liberazione memoria host
    free(height);
    free(excess);
    free(totalExcess);
    free(avq);

    return 0;
}
