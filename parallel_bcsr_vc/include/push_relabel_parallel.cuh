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

void preflow(int *capacity, int *excess, int *height, int *residual, int *totalExcess, int n, int s);

__global__ void pushKernel(int V, int source, int sink, int *d_height, int *d_excess, int *d_offset, int *d_column, int *d_capacities, int *d_flows, int *d_avq);

void globalRelabel(int *capacity, int *excess, int *height, int *residual, int *totalExcess, bool *scanned, bool *mark, int n, int t);

int pushRelabel(int *capacity, int *excess, int *height, int *residual, int *d_capacity, int *d_excess, int *d_height, int *d_residual, int *totalExcess, int n, int s, int t);

std::vector<int> findMinCutSetFromT(int n, int t, int *residual);

int executePushRelabel(std::string filename, std::string output);