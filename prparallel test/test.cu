#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <list>

using namespace std;

#define number_of_nodes V
#define number_of_edges E
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF 1000000000
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES V

void global_relabel(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, bool *mark, bool *scanned)
{
    for(int u = 0; u < V; u++)
    {
        for(int v = 0; v < V; v++)
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
        for(int i = 0; i < V; i++)
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

            for(y = 0; y < V; y++)
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

        for(int i = 0; i < V; i++)
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
            for(int i = 0; i < V; i++)
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

void preflow(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total)
{
    // initialising height values and excess flow, Excess_total values
    for(int i = 0; i < V; i++)
    {
        cpu_height[i] = 0; 
        cpu_excess_flow[i] = 0;
    }
    
    cpu_height[source] = V;
    *Excess_total = 0;

    // pushing flow in all edges going out from the source node
    for(int i = 0; i < V; i++)
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

__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx,int *gpu_rflowmtx)
{
    // u'th node is operated on by the u'th thread
    unsigned int u = (blockIdx.x*blockDim.x) + threadIdx.x;

    //printf("u : %d\nV : %d\n",u,V);

    if(u < V)
    {
        //printf("Thread id : %d\n",u);
        // cycle value is set to KERNEL_CYCLES as required 
        int cycle = KERNEL_CYCLES;

        /* Variables declared to be used inside the kernel :
        * e_dash - initial excess flow of node u
        * h_dash - height of lowest neighbor of node u
        * h_double_dash - used to iterate among height values to find h_dash
        * v - used to iterate among nodes to find v_dash
        * v_dash - lowest neighbor of node u 
        * d - flow to be pushed from node u
        */

        int e_dash,h_dash,h_double_dash,v,v_dash,d;

        while(cycle > 0)
        {
            if( (gpu_excess_flow[u] > 0) && (gpu_height[u] < V) )
            {
                e_dash = gpu_excess_flow[u];
                h_dash = INF;
                v_dash = NULL;

                for(v = 0; v < V; v++)
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

                if(gpu_height[u] > h_dash)
                {
                    /* height of u > height of lowest neighbor
                    * Push operation can be performed from node u to lowest neighbor
                    * All addition, subtraction and minimum operations are done using Atomics
                    * This is to avoid anomalies in conflicts between multiple threads
                    */

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
                }

                else
                {
                    /* height of u <= height of lowest neighbor,
                    * No neighbor with lesser height exists
                    * Push cannot be performed to any neighbor
                    * Hence, relabel operation is performed
                    */

                    gpu_height[u] = h_dash + 1;
                }

            }

            // cycle value is decreased
            cycle = cycle - 1;

        }
    }
}

void push_relabel(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx, int *gpu_rflowmtx)
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
    mark = (bool*)malloc(V*sizeof(bool));
    scanned = (bool*)malloc(V*sizeof(bool));

    // initialising mark values to false for all nodes
    for(int i = 0; i < V; i++)
    {
        mark[i] = false;
    }

    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        // copying height values to CUDA device global memory
        cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);

        printf("Invoking kernel\n");

        // invoking the push_relabel_kernel
        push_relabel_kernel<<<number_of_blocks_nodes,threads_per_block>>>(V,source,sink,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);

        cudaDeviceSynchronize();


        // copying height, excess flow and residual flow values from device to host memory
        cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_rflowmtx,gpu_rflowmtx,V*V*sizeof(int),cudaMemcpyDeviceToHost);

        printf("After invoking\n");
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
        // perform the global_relabel routine on host
        global_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,mark,scanned);

        printf("\nAfter global relabel\n");
        //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
    }

}

void print(int V,int *cpu_height, int *cpu_excess_flow, int *cpu_rflowmtx, int *cpu_adjmtx)
{
    printf("\nHeight :");
    for(int i = 0; i < V; i++)
    {
        printf("%d ",cpu_height[i]);
    }

    printf("\nExcess flow :");
    for(int i = 0; i < V; i++)
    {
        printf("%d ",cpu_excess_flow[i]);
    }

    printf("\nRflow mtx :\n");
    for(int i = 0; i < V; i++)
    {
        for(int j = 0; j < V; j++)
        {
            printf("%d ", cpu_rflowmtx[IDX(i,j)]);
        }
        printf("\n");
    }

    printf("\nAdj mtx :\n");
    for(int i = 0; i < V; i++)
    {
        for(int j = 0; j < V; j++)
        {
            printf("%d ", cpu_adjmtx[IDX(i,j)]);
        }
        printf("\n");
    }
}


void readgraph(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx)
{
    // initialising all adjacent matrix values to 0 before input 
    for(int i = 0; i < (number_of_nodes)*(number_of_nodes); i++)
    {
        cpu_adjmtx[i] = 0;
        cpu_rflowmtx[i] = 0;
    }
    // declaring file pointer to read edgelist
    FILE *fp = fopen("edgelist.txt","r");

    // declaring variables to read and store data from file
    char buf1[10],buf2[10],buf3[10];
    int e1,e2,cp;

    // getting edgelist input from file "edgelist.txt"
    for(int i = 0; i < E; i++)
    {
        // reading from file
        fscanf(fp,"%s",buf1);
        fscanf(fp,"%s",buf2);
        fscanf(fp,"%s",buf3);

        // storing as integers
        e1 = atoi(buf1);
        e2 = atoi(buf2);
        cp = atoi(buf3);

        /* Adding edges to the graph 
         * rflow - residual flow is also updated simultaneously
         * So the graph when prepared already has updated residual flow values
         * This is why residual flow is not initialised during preflow
         */

            cpu_adjmtx[IDX(e1,e2)] = cp;
            cpu_rflowmtx[IDX(e1,e2)] = cp;    
        

    }

}


int main(int argc, char **argv)
{
    // checking if sufficient number of arguments (4) are passed in CLI
    if(argc != 5)
    {
        printf("Invalid number of arguments passed during execution\n");
        exit(0);
    }

    // reading the arguments passed in CLI
    int V = atoi(argv[1]);
    int E = atoi(argv[2]);
    int source = atoi(argv[3]);
    int sink = atoi(argv[4]);

    // declaring variables to store graph data on host as well as on CUDA device global memory 
    int *cpu_height,*gpu_height;
    int *cpu_excess_flow,*gpu_excess_flow;
    int *Excess_total;
    int *cpu_adjmtx,*gpu_adjmtx;
    int *cpu_rflowmtx,*gpu_rflowmtx;
    
    // allocating host memory
    cpu_height = (int*)malloc(V*sizeof(int));
    cpu_excess_flow = (int*)malloc(V*sizeof(int));
    cpu_adjmtx = (int*)malloc(V*V*sizeof(int));
    cpu_rflowmtx = (int*)malloc(V*V*sizeof(int));
    Excess_total = (int*)malloc(sizeof(int));

    // allocating CUDA device global memory
    cudaMalloc((void**)&gpu_height,V*sizeof(int));
    cudaMalloc((void**)&gpu_excess_flow,V*sizeof(int));
    cudaMalloc((void**)&gpu_adjmtx,V*V*sizeof(int));
    cudaMalloc((void**)&gpu_rflowmtx,V*V*sizeof(int));

    // readgraph
    readgraph(V,E,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx);
    cpu_adjmtx[0] = 0;
    cpu_adjmtx[1] = 2;
    cpu_adjmtx[2] = 4;
    cpu_adjmtx[3] = 0;
    cpu_adjmtx[4] = 0;
    cpu_adjmtx[5] = 0;
    cpu_adjmtx[6] = 3;
    cpu_adjmtx[7] = 1;
    cpu_adjmtx[8] = 0;
    cpu_adjmtx[9] = 0;
    cpu_adjmtx[10] = 0;
    cpu_adjmtx[11] = 5;
    cpu_adjmtx[12] = 0;
    cpu_adjmtx[13] = 0;
    cpu_adjmtx[14] = 0;
    cpu_adjmtx[15] = 0;
    cpu_rflowmtx[0] = 0;
    cpu_rflowmtx[1] = 2;
    cpu_rflowmtx[2] = 4;
    cpu_rflowmtx[3] = 0;
    cpu_rflowmtx[4] = 0;
    cpu_rflowmtx[5] = 0;
    cpu_rflowmtx[6] = 3;
    cpu_rflowmtx[7] = 1;
    cpu_rflowmtx[8] = 0;
    cpu_rflowmtx[9] = 0;
    cpu_rflowmtx[10] = 0;
    cpu_rflowmtx[11] = 5;
    cpu_rflowmtx[12] = 0;
    cpu_rflowmtx[13] = 0;
    cpu_rflowmtx[14] = 0;
    cpu_rflowmtx[15] = 0;

    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // time start

    // invoking the preflow function to initialise values in host
    preflow(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total);

    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_adjmtx,cpu_adjmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);

    // push_relabel()
    push_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);

    // print values from both implementations
    printf("The maximum flow value of this flow network as calculated by the parallel implementation is %d\n",cpu_excess_flow[sink]);

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