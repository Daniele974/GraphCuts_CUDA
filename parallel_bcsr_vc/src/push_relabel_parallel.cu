#include "../include/push_relabel_parallel.cuh"
#include <cooperative_groups.h>

//TODO: da spostare in include
#define threads_per_block 256
#define numBlocksPerSM 1
#define numThreadsPerBlock 1024

using namespace cooperative_groups;
namespace cg = cooperative_groups; 

__device__ unsigned int avqSize;

void initialize(int V, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, int *totalExcess){
    for (int i = 0; i < V; i++){
        height[i] = 0;
        excess[i] = 0;
    }
    height[source] = V;
    *totalExcess = 0;

    for(int i = offset[source]; i < offset[source+1]; i++){
        int neighbor = column[i];
        if(capacities[i] > 0){
            forwardFlow[i] = 0;
            // ciclo per gestire archi da vicini di source a source
            for(int j = offset[neighbor]; j < offset[neighbor+1]; j++){
                if(column[j] == source){
                    forwardFlow[j] = capacities[i];
                    break;
                }
            }
            excess[neighbor] = capacities[i];
            *totalExcess = *totalExcess + excess[neighbor];
        }else{
            continue;
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
    for(int u = idx; u < V; u += blockDim.x * gridDim.x){
        if(d_excess[u] > 0 && d_height[u] < V && u != source && u != sink){
            int pos = atomicAdd(&avqSize, 1);
            d_avq[pos] = u;
        }
    }
}

template <unsigned int tileSize> __device__  int tiledSearchNeighbor(thread_block_tile <tileSize> tile, int pos, int *s_height, int *s_vid, int *s_vidx, int *Vindex, int V, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq){
    unsigned int idx = tile.thread_rank();
    //int tileId = threadIdx.x / tileSize;
    int u = d_avq[pos];
    int degree = d_offset[u+1] - d_offset[u];
    int numIters = (int)ceilf((float)degree / (float)tileSize);

    int minH = INF;
    int minV = -1;

    s_height[threadIdx.x] = INF;
    s_vid[threadIdx.x] = -1;
    s_vidx[threadIdx.x] = -2;
    tile.sync();

    for(int i = 0; i < numIters; i++){
        int vPos, v;
        if(i*tileSize + idx < degree){
            vPos = d_offset[u] + i*tileSize + idx;
            v = d_column[vPos];
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
        if(idx == 0){
            if(minH > s_height[threadIdx.x]){
                minH = s_height[threadIdx.x];
                minV = s_vid[threadIdx.x];
                *Vindex = s_vidx[threadIdx.x];
            }
        }
        tile.sync();
        s_height[threadIdx.x] = INF;
        s_vid[threadIdx.x] = -1;
        tile.sync();
    }
    tile.sync();
    return minV;
}

__global__ void pushKernel(int V, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq, int *d_cycle){
    grid_group grid = this_grid();
    thread_block block = this_thread_block();
    const int tileSize = 32;
    thread_block_tile<tileSize> tile = tiled_partition<tileSize>(block);
    int numTilesPerBlock = (blockDim.x + tileSize - 1) / tileSize;
    int numTilesPerGrid = numTilesPerBlock * gridDim.x;
    int tileIdx = blockIdx.x * numTilesPerBlock + block.thread_rank() / tileSize;
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int minV = -1;
    int Vindex = -1;
    int cycle = V;

    extern __shared__ int shared[];
    int *s_height = shared;
    int *s_vid = (int *)&shared[blockDim.x];
    int *s_vidx = (int *)&s_vid[blockDim.x];

    while(cycle > 0){
        scanActiveVertices(V, source, sink, d_height, d_excess, d_avq);
        grid.sync();

        if(avqSize == 0){
            break;
        }
        
        Vindex = -1;
        grid.sync();

        for(int i = tileIdx; i < avqSize; i += numTilesPerGrid){
            int u = d_avq[i];
            minV = tiledSearchNeighbor<tileSize>(tile, i, s_height, s_vid, s_vidx, &Vindex, V, source, sink, d_height, d_excess, d_offset, d_column, d_capacities, d_flows, d_avq);
            tile.sync();
            
            if(tile.thread_rank() == 0){
                if(minV == -1){
                    d_height[u] = V;
                }else{
                    if(d_height[u] > d_height[minV]){
                        int d;
                        int backwardIdx = -1;
                        for(int j = d_offset[minV]; j < d_offset[minV+1]; j++){
                            if(d_column[j] == u){
                                backwardIdx = j;
                                break;
                            }
                        }
                        if(backwardIdx == -1){
                            printf("Error: backward edge not found\n");
                            return;
                        }
                        if(d_excess[u] > d_flows[Vindex]){
                            d = d_flows[Vindex];
                        }else{
                            d = d_excess[u];
                        }
                        atomicAdd(&d_flows[backwardIdx], d);
                        atomicSub(&d_flows[Vindex], d);
                        atomicAdd(&d_excess[minV], d);
                        atomicSub(&d_excess[u], d);
                    }else{
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
    if(idx == 0){
        d_status[sink] = 1;
        *d_queueSize = 1;
        *d_level = 1;
        d_queue[0] = sink;
    }
    
    grid.sync();

    if(idx == 0){
        printf("Status: ");
        for(int i = 0; i < V; i++){
            printf(" %d,", d_status[i]);
        }
        printf("\n");

        printf("Queue: ");
        for(int i = 0; i < *d_queueSize; i++){
            printf(" %d,", d_queue[i]);
        }
        printf("\n");

        printf("Level: %d\n", *d_level);

        printf("Total excess: %d\n", *d_totalExcess);
    }

    grid.sync();

    while(true){

        for (int i = idx; i < V; i += blockDim.x * gridDim.x) {
            if (d_status[i] == 0) {
                d_queue[atomicAdd(d_queueSize, 1)] = i;
            }
        }

        grid.sync();

        if(idx == 0){
            printf("DOPO AGGIUNTA FRONTIERA A CODA\n");
            printf("\tStatus: ");
            for(int i = 0; i < V; i++){
                printf(" %d,", d_status[i]);
            }
            printf("\n");

            printf("\tQueue: ");
            for(int i = 0; i < *d_queueSize; i++){
                printf(" %d,", d_queue[i]);
            }
            printf("\n");

            printf("\tExcess: ");
            for(int i = 0; i < V; i++){
                printf(" %d,", d_excess[i]);
            }
            printf("\n");

            printf("\tFlow: ");
            for(int i = 0; i < E; i++){
                printf(" %d,", d_flows[i]);
            }
            printf("\n");

            printf("\tLevel: %d\n", *d_level);

            printf("\tTotal excess: %d\n", *d_totalExcess);
        }

        grid.sync();

        for(int i = idx; i < *d_queueSize; i+= blockDim.x * gridDim.x){
            int u = d_queue[i];
            for(int j = d_offset[u]; j < d_offset[u+1]; j++){
                int v = d_column[j];
                if(d_status[v] < 0 && d_flows[j] > 0){
                    d_status[v] = 0;
                    d_height[v] = *d_level + 1;
                    *terminate = false;
                    break;
                }
            }
        }
        
        grid.sync();

        if(idx == 0){
            printf("DOPO ELABORAZIONE CODA\n");
            printf("\tStatus: ");
            for(int i = 0; i < V; i++){
                printf(" %d,", d_status[i]);
            }
            printf("\n");

            printf("\tQueue: ");
            for(int i = 0; i < *d_queueSize; i++){
                printf(" %d,", d_queue[i]);
            }
            printf("\n");

            printf("\tExcess: ");
            for(int i = 0; i < V; i++){
                printf(" %d,", d_excess[i]);
            }
            printf("\n");

            printf("\tLevel: %d\n", *d_level);

            printf("\tTotal excess: %d\n", *d_totalExcess);
        }

        grid.sync();

        if(*terminate){
            break;
        }
        
        grid.sync();        

        if(idx == 0){
            *d_queueSize = 0;
            *d_level = *d_level + 1;
            *terminate = true;
        }

        grid.sync();

        /*for (int i = idx; i < V; i += blockDim.x * gridDim.x) {
            if (d_status[i] == -1) {
                d_queue[atomicAdd(d_queueSize, 1)] = i;
            }
        }
        grid.sync();

        for(int i = idx; i < *d_queueSize; i+= blockDim.x * gridDim.x){
            int u = d_queue[i];
            for(int j = d_offset[u]; j < d_offset[u+1]; j++){
                int v = d_column[j];
                if(d_status[v] >= 0 && d_flows[j] > 0){
                    d_status[u] = *d_level + 1;
                    d_height[u] = d_status[u];
                    *terminate = false;
                    break;
                }
            }
        }
        grid.sync();
        if(*terminate){
            break;
        }
        grid.sync();
        if(idx == 0){
            *d_queueSize = 0;
            *d_level = *d_level + 1;
            *terminate = true;
        }
        grid.sync();
        */
    }
    grid.sync();

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
    int *d_status, *d_queue, *d_level, *d_totalExcess, *d_queueSize;
    bool *terminate;

    cudaMalloc((void**)&d_status, V*sizeof(int));
    cudaMalloc((void**)&d_queue, V*sizeof(int));
    cudaMalloc((void**)&d_level, sizeof(int));
    cudaMalloc((void**)&d_totalExcess, sizeof(int));
    cudaMalloc((void**)&d_queueSize, sizeof(int));
    cudaMalloc((void**)&terminate, sizeof(bool));

    cudaMemset(d_status, -1, V*sizeof(int));
    cudaMemset(d_queue, 0, V*sizeof(int));
    cudaMemset(d_level, 0, sizeof(int));
    cudaMemset(d_queueSize, 0, sizeof(int));
    cudaMemset(terminate, true, sizeof(bool));

    cudaMemcpy(d_height, height, V*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_excess, excess, V*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flows, forwardFlow, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_totalExcess, totalExcess, sizeof(int), cudaMemcpyHostToDevice);

    //Configurazione GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    dim3 num_blocks(prop.multiProcessorCount * numBlocksPerSM);
    dim3 block_size(numThreadsPerBlock);  

    void *kernel_args[] = {&V, &E, &source, &sink, &d_height, &d_excess, &d_offset, &d_column, &d_capacities, &d_flows, &d_status, &d_queue, &d_queueSize, &d_level, &d_totalExcess, &terminate};

    cudaError_t cudaStatus;
    cudaStatus = cudaLaunchCooperativeKernel((void*)globalRelabelKernel, num_blocks, block_size, kernel_args, 0, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error, for example, by cleaning up resources and exiting
        exit(1);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(height, d_height, V*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(excess, d_excess, V*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(forwardFlow, d_flows, E*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(totalExcess, d_totalExcess, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_status);
    cudaFree(d_queue);
    cudaFree(d_level);
    cudaFree(d_totalExcess);
    cudaFree(d_queueSize);
    cudaFree(terminate);
}

void pushRelabel(int V, int E, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, int *totalExcess, 
                int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq, int *d_cycle){
    bool *mark, *scanned;
    mark = (bool *)malloc(V*sizeof(bool));
    scanned = (bool *)malloc(V*sizeof(bool));

    //Configurazione GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    dim3 num_blocks(prop.multiProcessorCount * numBlocksPerSM);
    dim3 block_size(numThreadsPerBlock);  

    size_t sharedMemSize = 3 * block_size.x * sizeof(int);

    //void* original_kernel_args[] = {&V, &source, &sink, &d_height, &d_excess, &d_offset, &d_column, &d_capacities, &d_flows};

    void* kernel_args[] = {&V, &source, &sink, &d_height, &d_excess, 
                        &d_offset, &d_column, &d_capacities, &d_flows, &d_avq, &d_cycle}; 
    
    for(int i = 0; i < V; i++){
        mark[i] = false;
    }

    while(excess[source] + excess[sink] < *totalExcess){
        cudaMemcpy(d_height, height, V*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_excess, excess, V*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flows, forwardFlow, E*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_cycle, V, sizeof(int));
        
        cudaError_t cudaStatus;
        cudaStatus = cudaLaunchCooperativeKernel((void*)pushKernel, num_blocks, block_size, kernel_args, sharedMemSize, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error, for example, by cleaning up resources and exiting
            exit(1);
        }
        cudaDeviceSynchronize();

        cudaMemcpy(height, d_height, V*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(excess, d_excess, V*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(forwardFlow, d_flows, E*sizeof(int), cudaMemcpyDeviceToHost);

        globalRelabel(V, E, source, sink, height, excess, offset, column, capacities, forwardFlow, d_height, d_excess, d_offset, d_column, d_capacities, d_flows, totalExcess, mark, scanned); 
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
    // Lettura grafo da file
    //readGraphFromFile(filename, n, &capacity);
    int V = 6;
    int E = 8;
    int source = 0;
    int sink = 5;

    int *height, *excess,  *totalExcess, *avq;
    int cycle = V;
    int *d_height, *d_excess, *d_column, *d_offset, *d_capacities, *d_flows, *d_avq, *d_cycle;

    height = (int *)malloc(V*sizeof(int));
    excess = (int *)malloc(V*sizeof(int));
    totalExcess = (int *)malloc(sizeof(int));
    avq = (int *)malloc(V*sizeof(int));
    
    for (int i = 0; i < V; i++)
    {
        avq[i] = 0;
    }

    cudaMalloc((void**)&d_height, V*sizeof(int));
    cudaMalloc((void**)&d_excess, V*sizeof(int));
    cudaMalloc((void**)&d_column, E*sizeof(int));
    cudaMalloc((void**)&d_offset, (V+1)*sizeof(int));
    cudaMalloc((void**)&d_capacities, E*sizeof(int));
    cudaMalloc((void**)&d_flows, E*sizeof(int));
    cudaMalloc((void**)&d_avq, V*sizeof(int));
    cudaMalloc((void**)&d_cycle, sizeof(int));
    
    /*
    int e_offset[] = {0,2,3,5,6,8,8};
    int e_column[] = {1,2,3,3,4,5,3,5}; //destinations
    int e_forwardFlow[] = {3,7,4,2,5,9,3,2};
    int e_offset[] = {0,0,1,2,5,6,8};
    int e_column[] = {0,0,1,2,4,2,3,4}; //destinations
    int e_forwardFlow[] = {0,0,0,0,0,0,0,0};
    int e_capacities[] = {3,7,4,2,5,9,3,2};
    */
    E = 2*E;
    int e_offset[] = {0,2,4,7,11,14,16};
    int e_column[] = {1,2,0,3,0,3,4,1,2,4,5,2,3,5,3,4};
    int e_capacities[] = {3,7,0,4,0,2,5,0,0,0,9,0,3,2,0,0};
    int e_forwardFlow[] = {3,7,0,4,0,2,5,0,0,0,9,0,3,2,0,0};

    initialize(V, source, sink, height, excess, e_offset, e_column, e_capacities, e_forwardFlow, totalExcess);

    printf("\n");
    /*printf("Total excess: %d\n", *totalExcess);
    
    for (int i = 0; i < V; i++)
    {
        printf("Excess[%d]: %d\n", i, excess[i]);
    }
    
    for (int i = 0; i < E; i++)
    {
        printf("ForwardFlow[%d]: %d\n", i, e_forwardFlow[i]);
    }

    for (int i = 0; i < V; i++)
    {
        printf("Height[%d]: %d\n", i, height[i]);
    }
    */
    


    cudaMemcpy(d_height, height, V*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_excess, excess, V*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, e_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column, e_column, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacities, e_capacities, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flows, e_forwardFlow, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_avq, avq, V*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cycle, &cycle, sizeof(int), cudaMemcpyHostToDevice);

    // PUSH RELABEL QUI
    pushRelabel(V, E, source, sink, height, excess, e_offset, e_column, e_capacities, e_forwardFlow, totalExcess, d_height, d_excess, d_offset, d_column, d_capacities, d_flows, d_avq, d_cycle);
    
    /*
    printf("Total excess: %d\n", *totalExcess);
    for (int i = 0; i < V; i++)
    {
        printf("Excess[%d]: %d\n", i, excess[i]);
    }
    */

    cudaFree(d_height);
    cudaFree(d_excess);
    cudaFree(d_offset);
    cudaFree(d_column);
    cudaFree(d_capacities);
    cudaFree(d_flows);
    cudaFree(d_avq);
    cudaFree(d_cycle);

    free(height);
    free(excess);
    free(totalExcess);
    free(avq);

    return 0;
}
