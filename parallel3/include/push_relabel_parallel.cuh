#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <list>
#include <vector>
#include <queue>

#include <iostream>
#include <string.h>

#include "utils.hpp"
#include "file_manager.hpp"

#define _DEBUG 0

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
 
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


//#define number_of_nodes *V
//#define number_of_edges *E
//#define threads_per_block 256
//#define numBlocksPerSM 1
//#define numThreadsPerBlock 1024
//#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
//#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF INT_MAX
//#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
//#define KERNEL_CYCLES V
//#define ull unsigned long long

void preflow(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess);
__global__ void push_relabel_kernel(int V, int source, int sink, int *d_capacities, int *d_residual, int *d_height,int *d_excess);
void global_relabel(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess, bool *mark, bool *scanned);
void push_relabel(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess, int *d_capacities, int *d_residual, int *d_height, int *d_excess);
int executePushRelabel(std::string filename, std::string output);