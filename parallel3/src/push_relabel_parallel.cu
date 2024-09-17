#include "../include/push_relabel_parallel.cuh"

using namespace cooperative_groups;

void preflow(int *V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total)
{
    // initialising height values and excess flow, Excess_total values
    for(int i = 0; i < *V; i++)
    {
        cpu_height[i] = 0; 
        cpu_excess_flow[i] = 0;
    }
    
    cpu_height[source] = *V;
    *Excess_total = 0;

    // pushing flow in all edges going out from the source node
    for(int i = 0; i < *V; i++)
    {
        // for all (source,i) belonging to E :
        if(cpu_adjmtx[IDX(source,i)] > 0)
        {
            // pushing out of source node
            cpu_rflowmtx[IDX(source,i)] = 0;
            
            /* updating the residual flow value on the back edge
             * u_f(x,s) = u_xs + u_sx
             * The capacity of the back edge is also added to avoid any push operation back to the source 
             * This avoids creating a race condition, where flow keeps travelling to and from the source
             */
            cpu_rflowmtx[IDX(i,source)] = cpu_adjmtx[IDX(source,i)] + cpu_adjmtx[IDX(i,source)];
            
            // updating the excess flow value of the node flow is pushed to, from the source
            cpu_excess_flow[i] = cpu_adjmtx[IDX(source,i)];

            // update Excess_total value with the new excess flow value of the node flow is pushed to
            *Excess_total += cpu_excess_flow[i];
        } 
    }

}

__global__ void push_relabel_kernel(int *V, int source, int sink, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx,int *gpu_rflowmtx)
{
    grid_group grid = this_grid();
    // u'th node is operated on by the u'th thread
    unsigned int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    // printf("thread id : %d\n",idx);

    int cycle = *V;

    while (cycle > 0) {

        for (unsigned int u = idx; u < *V; u += blockDim.x * gridDim.x) {

            //printf("Thread id : %d\n",u);

            /* Variables declared to be used inside the kernel :
            * e_dash - initial excess flow of node u
            * h_dash - height of lowest neighbor of node u
            * h_double_dash - used to iterate among height values to find h_dash
            * v - used to iterate among nodes to find v_dash
            * v_dash - lowest neighbor of node u 
            * d - flow to be pushed from node u
            */

            int e_dash, h_dash, h_double_dash, d;
            int v,v_dash;

            if( (gpu_excess_flow[u] > 0) && (gpu_height[u] < *V) && u != source && u != sink)
            {
                e_dash = gpu_excess_flow[u];
                h_dash = INF;
                v_dash = -1;

                for(v = 0; v < *V; v++)
                {
                    // for all (u,v) belonging to E_f (residual graph edgelist)
                    if(gpu_rflowmtx[IDX(u,v)] > 0)
                    {
                        h_double_dash = gpu_height[v];
                        // finding lowest neighbor of node u
                        if(h_double_dash < h_dash)
                        {
                            v_dash = v;
                            h_dash = h_double_dash;
                        }
                    }
                }
                if (v_dash == -1) {
                    gpu_height[u] = *V;
                } else {
                    if(gpu_height[u] > h_dash)
                    {
                        /* height of u > height of lowest neighbor
                        * Push operation can be performed from node u to lowest neighbor
                        * All addition, subtraction and minimum operations are done using Atomics
                        * This is to avoid anomalies in conflicts between multiple threads
                        */
                        // printf("thread %d : Pushing from %d to %d\n", idx, u,v_dash);
                        // printf("-gpu_rflowmtx[IDX(v_dash,u)]: %d\n",gpu_rflowmtx[IDX(v_dash,u)]);
                        // printf("-gpu_excess_flow[u]: %d\n",gpu_excess_flow[u]);
                        // d captures flow to be pushed 
                        d = e_dash;
                        //atomicMin(&d,gpu_rflowmtx[IDX(u,v_dash)]);
                        if(e_dash > gpu_rflowmtx[IDX(u,v_dash)])
                        {
                            d = gpu_rflowmtx[IDX(u,v_dash)];
                        }
                        // Residual flow towards lowest neighbor from node u is increased
                        atomicAdd(&gpu_rflowmtx[IDX(v_dash,u)],d);

                        // Residual flow towards node u from lowest neighbor is decreased
                        atomicSub(&gpu_rflowmtx[IDX(u,v_dash)],d);

                        // Excess flow of lowest neighbor and node u are updated
                        atomicAdd(&gpu_excess_flow[v_dash],d);
                        atomicSub(&gpu_excess_flow[u],d);
                        // printf("*gpu_rflowmtx[IDX(v_dash,u)]: %d\n",gpu_rflowmtx[IDX(v_dash,u)]);
                        // printf("*gpu_excess_flow[u]: %d\n",gpu_excess_flow[u]);
                    }

                    else
                    {
                        /* height of u <= height of lowest neighbor,
                        * No neighbor with lesser height exists
                        * Push cannot be performed to any neighbor
                        * Hence, relabel operation is performed
                        */
                        // printf("thread %d : Relabeling %d\n", idx, u);
                        // printf("-gpu_height[u]: %d\n",gpu_height[u]);
                        gpu_height[u] = h_dash + 1;
                    }
                }
            }
        }
        // cycle value is decreased
        cycle = cycle - 1;
        grid.sync();
    }
}

void global_relabel(int *V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, bool *mark, bool *scanned)
{
    for(int u = 0; u < *V; u++)
    {
        for(int v = 0; v < *V; v++)
        {
            // for all (u,v) belonging to E
            if(cpu_adjmtx[IDX(u,v)] > 0)
            {
                if(cpu_height[u] > cpu_height[v] + 1)
                {
                    cpu_excess_flow[u] = cpu_excess_flow[u] - cpu_rflowmtx[IDX(u,v)];
                    cpu_excess_flow[v] = cpu_excess_flow[v] + cpu_rflowmtx[IDX(u,v)];
                    cpu_rflowmtx[IDX(v,u)] = cpu_rflowmtx[IDX(v,u)] + cpu_rflowmtx[IDX(u,v)];
                    cpu_rflowmtx[IDX(u,v)] = 0;
                }
            }
        }
    }
        printf("Prebfs\n");
        // performing backwards bfs from sink and assigning height values with each vertex's BFS tree level
        
        // declaring the Queue 
        std::list<int> Queue;

        // declaring variables to iterate over nodes for the backwards bfs and to store current tree level
        int x,y,current;
        
        // initialisation of the scanned array with false, before performing backwards bfs
        for(int i = 0; i < *V; i++)
        {
            scanned[i] = false;
        }

        // Enqueueing the sink and set scan(sink) to true 
        Queue.push_back(sink);
        scanned[sink] = true;
        cpu_height[sink] = 0;
        // bfs routine and assigning of height values with tree level values
        while(!Queue.empty())
        {
            // dequeue
            x = Queue.front();
            Queue.pop_front();

            // capture value of current level
            current = cpu_height[x];
            
            // increment current value
            current = current + 1;

            for(y = 0; y < *V; y++)
            {
                // for all (y,x) belonging to E_f (residual graph)
                if(cpu_rflowmtx[IDX(y,x)] > 0)
                {
                    // if y is not scanned
                    if(scanned[y] == false)
                    {
                        // assign current as height of y node
                        cpu_height[y] = current;

                        // mark scanned(y) as true
                        scanned[y] = true;

                        // Enqueue y
                        Queue.push_back(y);
                    }
                }
            }

        }
        printf("Pre check\n");
        // declaring and initialising boolean variable for checking if all nodes are relabeled
        bool if_all_are_relabeled = true;

        for(int i = 0; i < *V; i++)
        {
            if(scanned[i] == false)
            {
                if_all_are_relabeled = false;
                break;
            }
        }

        // if not all nodes are relabeled
        if(if_all_are_relabeled == false)
        {
            // for all nodes
            for(int i = 0; i < *V; i++)
            {
                // if i'th node is not marked or relabeled
                if( !( (scanned[i] == true) || (mark[i] == true) ) )
                {
                    // mark i'th node
                    mark[i] = true;

                    /* decrement excess flow of i'th node from Excess_total
                     * This shows that i'th node is not scanned now and needs to be marked, thereby no more contributing to Excess_total
                     */

                    *Excess_total = *Excess_total - cpu_excess_flow[i];
                }
            }
        }

}

void push_relabel(int *V, int *gpu_V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx, int *gpu_rflowmtx)
{
    /* Instead of checking for overflowing vertices(as in the sequential push relabel),
     * sum of excess flow values of sink and source are compared against Excess_total 
     * If the sum is lesser than Excess_total, 
     * it means that there is atleast one more vertex with excess flow > 0, apart from source and sink
     */

    /* declaring the mark and scan boolean arrays used in the global_relabel routine outside the while loop 
     * This is not to lose the mark values if it goes out of scope and gets redeclared in the next iteration 
     */

    bool *mark,*scanned;
    mark = (bool*)malloc(*V*sizeof(bool));
    scanned = (bool*)malloc(*V*sizeof(bool));

    // Set the timer
    //CudaTimer timer;
    //float totalMilliseconds = 0.0f;

    // Configure the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    dim3 num_blocks(deviceProp.multiProcessorCount * numBlocksPerSM);
    dim3 block_size(numThreadsPerBlock);

     // Print GPU device name
    printf("GPU Device: %s\n", deviceProp.name);
    printf("Number of blocks: %d\n", num_blocks.x);
    printf("Number of threads per block: %d\n", block_size.x);

    void* kernel_args[] = {&gpu_V,&source,&sink,&gpu_height,&gpu_excess_flow,&gpu_adjmtx,&gpu_rflowmtx};


    // initialising mark values to false for all nodes
    for(int i = 0; i < *V; i++)
    {
        mark[i] = false;
    }

    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        // printf("cpu_excess_flow[source]: %d, cpu_excess_flow[sink]: %d\n",cpu_excess_flow[source], cpu_excess_flow[sink]);
        // copying height values to CUDA device global memory
        CHECK(cudaMemcpy(gpu_height,cpu_height,*V*sizeof(int),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_excess_flow, cpu_excess_flow, *V*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,*V**V*sizeof(int),cudaMemcpyHostToDevice));

        printf("Invoking kernel\n");

        //timer.start();
        cudaError_t cudaStatus;
        // invoking the push_relabel_kernel
        // push_relabel_kernel<<<num_blocks,block_size>>>(gpu_V,source,sink,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);

        cudaStatus = cudaLaunchCooperativeKernel((void*)push_relabel_kernel, num_blocks, block_size, kernel_args, 0, 0);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error, for example, by cleaning up resources and exiting
            exit(1);
        }

        cudaDeviceSynchronize();
        //timer.stop();
        //totalMilliseconds += timer.elapsed();

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            // Handle error
            fprintf(stderr, "Kernel launching error: %s\n", cudaGetErrorString(error));
        }

        // copying height, excess flow and residual flow values from device to host memory
        CHECK(cudaMemcpy(cpu_height,gpu_height,*V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_excess_flow,gpu_excess_flow,*V*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cpu_rflowmtx,gpu_rflowmtx,*V**V*sizeof(int),cudaMemcpyDeviceToHost));
        
        printf("After invoking\n");
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);

        // perform the global_relabel routine on host
        global_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,mark,scanned);

        printf("\nAfter global relabel\n");
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
    }

    //printf("Total kernel time: %.6f ms\n", totalMilliseconds);
}

int executePushRelabel(std::string filename, std::string output)
{
    // Parse command line arguments
    


    int *V = new int(0);
    int *gpu_V;
    int *E = new int(0);
    int *source = new int(0);
    int *sink = new int(0);
    
    // declaring variables to store graph data on host as well as on CUDA device global memory 
    int *cpu_height = NULL,*gpu_height = NULL;
    int *cpu_excess_flow = NULL,*gpu_excess_flow = NULL;
    int *Excess_total = NULL;
    int *cpu_adjmtx = NULL,*gpu_adjmtx = NULL;
    int *cpu_rflowmtx = NULL,*gpu_rflowmtx = NULL;

    readGraphFromFile(filename, *V, &cpu_adjmtx);
    printf("V: %d\n", *V);
    
    *source = 0;
    *sink = *V - 1;

    // allocating memory for residual flow matrix
    cpu_rflowmtx = (int*)malloc(*V**V*sizeof(int));

    for(int i = 0; i < *V; i++){
        for(int j = 0; j < *V; j++){
            cpu_rflowmtx[IDX(i,j)] = cpu_adjmtx[IDX(i,j)];
            if(cpu_adjmtx[IDX(i,j)] > 0){
                *E = *E + 1;
            }
        }
    }

    printf("E: %d\n", *E);


    
    // Print the graph's information
    //printf("Reading graph from %s\n", filename->c_str());
    printf("Number of vertices: %d\n", *V);
    printf("Number of edges: %d\n", *E);
    printf("Source vertex: %d\n", *source);
    printf("Sink vertex: %d\n", *sink);
    
    // allocating host memory
    cpu_height = (int*)malloc(*V*sizeof(int));
    cpu_excess_flow = (int*)malloc(*V*sizeof(int));
    Excess_total = (int*)malloc(sizeof(int));
    


    // allocating CUDA device global memory
    CHECK(cudaMalloc((void**)&gpu_height,*V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_excess_flow,*V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_adjmtx,*V**V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_rflowmtx,*V**V*sizeof(int)));
    CHECK(cudaMalloc((void**)&gpu_V,sizeof(int)));


    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // time start

    // invoking the preflow function to initialise values in host
    preflow(V,*source,*sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total);

    // print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    CHECK(cudaMemcpy(gpu_height,cpu_height,*V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_excess_flow,cpu_excess_flow,*V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_adjmtx,cpu_adjmtx,*V**V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,*V**V*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_V,V,sizeof(int),cudaMemcpyHostToDevice));

    // push_relabel()
    push_relabel(V,gpu_V,*source,*sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);
    

    // print values from both implementations
    printf("The maximum flow value of this flow network as calculated by the parallel implementation is %d\n",cpu_excess_flow[*sink]);
    

    // free device memory
    cudaFree(gpu_height);
    cudaFree(gpu_excess_flow);
    cudaFree(gpu_adjmtx);
    cudaFree(gpu_rflowmtx);
    
    // free host memory
    free(cpu_height);
    free(cpu_excess_flow);
    free(cpu_adjmtx);
    free(cpu_rflowmtx);
    free(Excess_total);
    
    // return 0 and end program
    return 0;

}