#include "../include/push_relabel_parallel.cuh"

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
 
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
    
    std::list<int> queue;
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

        std::cout << "Source excess: " << excess[s] << " Sink excess: " << excess[t] << " Total excess: " << *totalExcess << std::endl;

        if(_DEBUG) std::cout << "Push..." << std::endl;
        //HANDLE_ERROR(cudaMemcpy(d_excess, excess, n*sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_height, height, n*sizeof(int), cudaMemcpyHostToDevice));
        //HANDLE_ERROR(cudaMemcpy(d_residual, residual, n*n*sizeof(int), cudaMemcpyHostToDevice));
        pushKernel<<<gridSize, blockSize>>>(d_capacity, d_excess, d_height, d_residual, n);
        cudaDeviceSynchronize();
        if(_DEBUG) std::cout << "Push done" << std::endl;

        HANDLE_ERROR(cudaMemcpy(excess, d_excess, n*sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(height, d_height, n*sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(residual, d_residual, n*n*sizeof(int), cudaMemcpyDeviceToHost));

        if(_DEBUG) printStatus(capacity, excess, height, residual, totalExcess, n);

        if(_DEBUG) std::cout << "Global relabel..." << std::endl;
        globalRelabel(capacity, excess, height, residual, totalExcess, scanned, mark, n, t);
        if(_DEBUG) std::cout << "Global relabel done" << std::endl;
        
        if(_DEBUG) printStatus(capacity, excess, height, residual, totalExcess, n);
        
    }
    return excess[t];
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
    int n = 0;
    int *capacity = nullptr;

    // Lettura grafo da file
    readGraphFromFile(filename, n, &capacity);

    const auto start = std::chrono::high_resolution_clock::now();

    int s = 0;
    int t = n-1;
    int *flow = (int *)malloc(n*n*sizeof(int));
    int *excess = (int *)malloc(n*sizeof(int));
    int *height = (int *)malloc(n*sizeof(int));
    int *residual = (int *)malloc(n*n*sizeof(int));
    int *totalExcess = (int *)malloc(sizeof(int));

    int *d_capacity, *d_excess, *d_height, *d_residual;    

    HANDLE_ERROR(cudaMalloc((void**)&d_capacity, n*n*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_excess, n*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_height, n*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_residual, n*n*sizeof(int)));

    initialize(capacity, excess, height, residual, totalExcess, n, s);

    if(_DEBUG) std::cout << "Initialization done" << std::endl;
    if(_DEBUG) printStatus(capacity, excess, height, residual, totalExcess, n);
    
    const auto endInitialization = std::chrono::high_resolution_clock::now();

    HANDLE_ERROR(cudaMemcpy(d_capacity, capacity, n*n*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_excess, excess, n*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_height, height, n*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_residual, residual, n*n*sizeof(int), cudaMemcpyHostToDevice));
    
    if(_DEBUG) std::cout << "Push relabel..." << std::endl;
    int maxFlow = pushRelabel(capacity, excess, height, residual, d_capacity, d_excess, d_height, d_residual, totalExcess, n, s, t);
    const auto end = std::chrono::high_resolution_clock::now();
    if(_DEBUG) std::cout << "Push relabel done" << std::endl;
    
    //std::cout<<"Max flow: "<<maxFlow<<std::endl;

    auto initializationTime = std::chrono::duration_cast<std::chrono::microseconds>(endInitialization - start);
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - endInitialization);
    auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::vector<int> minCut = findMinCutSetFromT(n, t, residual);
    writeResultsToFile(output, excess[t], minCut, initializationTime, executionTime, totalTime);

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
