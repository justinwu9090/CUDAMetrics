#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

// more standard libraries
#include <iostream>
#include <vector>
#include <map>

using namespace std;

__global__ void addKernel(int* c, const int* a, const int* b);
__host__ map < string, string> addSequentialWrapper(int size, bool checkErrors);
__host__ map < string, string> addKernelWrapper(int size, bool checkErrors);
__global__ void addKernel(int* c, const int* a, const int* b);
