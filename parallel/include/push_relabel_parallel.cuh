#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <list>
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>

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

#define INF 1000000000

void preflow(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess);

__global__ void pushRelabelKernel(int V, int source, int sink, int *d_capacities, int *d_residual, int *d_height,int *d_excess);

void globalRelabel(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess, bool *mark, bool *scanned);

void pushRelabel(int V, int source, int sink, int *capacities, int *residual, int *height, int *excess, int *totalExcess, int *d_capacities, int *d_residual, int *d_height, int *d_excess);

std::vector<int> findMinCutSetFromSinkOMP(int n, int t, int *residual);

int executePushRelabel(std::string filename, std::string output, bool computeMinCut);