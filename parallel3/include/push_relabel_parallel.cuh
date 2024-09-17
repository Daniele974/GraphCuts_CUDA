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

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__,                    \
              cudaGetErrorName(err), cudaGetErrorString(err));                 \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#define number_of_nodes *V
#define number_of_edges *E
#define threads_per_block 256
#define numBlocksPerSM 1
#define numThreadsPerBlock 1024
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF INT_MAX
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES V
#define ull unsigned long long

void preflow(int *V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total);
__global__ void push_relabel_kernel(int *V, int source, int sink, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx,int *gpu_rflowmtx);
void push_relabel(int *V, int *gpu_V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx, int *gpu_rflowmtx);
void global_relabel(int *V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, bool *mark, bool *scanned);
int executePushRelabel(std::string filename, std::string output);