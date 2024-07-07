// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>

// more standard libraries
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using namespace std;

__host__ void errCatch(cudaError_t);
cudaDeviceProp dumpDeviceProperties(bool printout);