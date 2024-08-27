#include <cuda_runtime.h>
#include <list>
#include <vector>
#include "utils.hpp"
#include "file_manager.hpp"

#define INF 1000000

#define _DEBUG 0

void initialize(int *capacity, int *excess, int *height, int *residual, int *totalExcess, int n, int s);

__global__ void pushKernel(int *d_capacity, int *d_excess, int *d_height, int *d_residual, int n);

void globalRelabel(int *capacity, int *excess, int *height, int *residual, int *totalExcess, bool *scanned, bool *mark, int n, int t);

int pushRelabel(int *capacity, int *excess, int *height, int *residual, int *d_capacity, int *d_excess, int *d_height, int *d_residual, int *totalExcess, int n, int s, int t);

int executePushRelabel(std::string filename, std::string output);