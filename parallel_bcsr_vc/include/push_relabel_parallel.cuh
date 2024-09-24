#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <list>
#include <vector>
#include <queue>
#include "utils.hpp"
#include "file_manager.hpp"

#define INF 1000000

#define _DEBUG 0

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
 
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

using namespace cooperative_groups;
namespace cg = cooperative_groups; 

void preflow(int V, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, int *totalExcess);

__device__ void scanActiveVertices(int V, int source, int sink, int *d_height, int *d_excess, int *d_avq);

template <unsigned int tileSize> __device__  int tiledSearchNeighbor(thread_block_tile <tileSize> tile, int pos, int *s_height, int *s_vid, int *s_vidx, int *Vindex, int V, int source, int sink, 
                                                                        int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq);

__global__ void pushKernel(int V, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq);

__global__ void globalRelabelKernel(int V, int E, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, 
                                    int *d_status, int *d_queue, int *d_queueSize, int *d_level, int *d_totalExcess, bool *terminate);

void globalRelabel(int V, int E, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, 
                    int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *totalExcess, bool *mark, bool *scanned);

int pushRelabel(int V, int E, int source, int sink, int *height, int *excess, int *offset, int *column, int *capacities, int *forwardFlow, int *totalExcess, 
                int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq);

std::vector<int> findMinCutSetFromSink(int V, int sink, int *offset, int *column, int *forwardFlow);

int executePushRelabel(std::string filename, std::string output, bool computeMinCut);