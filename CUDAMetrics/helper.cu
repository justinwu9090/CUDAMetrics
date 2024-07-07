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

// Catches errors returned from CUDA functions
__host__ void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}

// Returns the size in bytes of any type of vector
template<typename T>
size_t vBytes(const typename vector<T>& v) {
	return sizeof(T) * v.size();
}

cudaDeviceProp dumpDeviceProperties(bool printout = true)
{
	int deviceID;
	cudaDeviceProp props;

	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&props, deviceID);
	if (printout)
	{
		cout << "GPU: " << props.name << endl;
		cout << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor << endl;
		cout << "maxBlocksPerMultiProcessor: " << props.maxBlocksPerMultiProcessor << endl;
		cout << "multiProcessorCount: " << props.multiProcessorCount << endl;
	}

	return props;
}