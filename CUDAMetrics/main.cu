// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

// more standard libraries
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using namespace std;

// local
#include "helper.cuh"
#include "add.cuh"

int main()
{
	cudaDeviceProp props = dumpDeviceProperties(true);

	// theoretical max occupancy threads based on specific CUDA device properties. # SM's X Max Blocks/SM X maxThreads per SM
	int maxoccupancythreads = props.multiProcessorCount * props.maxBlocksPerMultiProcessor * props.maxThreadsPerMultiProcessor;
	cout << "maxoccupancythreads: " << maxoccupancythreads << endl;


	int arraySize = maxoccupancythreads;

	ofstream myfile;
	myfile.open("test.csv", std::ios_base::app);

	// Add vectors in parallel - call AddKernel Wrapper
	//vector<unsigned int> sizes = { arraySize * 4, arraySize * 64, arraySize * 128, arraySize * 256, arraySize * 512, arraySize * 1024, arraySize * 2048 };
	vector<int> sizes({ arraySize * 4, arraySize * 64, arraySize * 128, arraySize * 256, arraySize * 512, arraySize * 1024, arraySize * 2048 });
	//vector<int> sizes({ arraySize * 4, arraySize * 64});
	for (auto arrsize : sizes)
	{

		auto results = addKernelWrapper(arrsize, false);

		// add to results csv.
		myfile << results["name"] << "," << results["grid_dim"] << "," << results["block_dim"] << "," << results["array_size"] << "," << results["duration_us"] << "," << endl;

		results = addSequentialWrapper(arrsize, false);
		myfile << results["name"] << "," << results["grid_dim"] << "," << results["block_dim"] << "," << results["array_size"] << "," << results["duration_us"] << "," << endl;
	}
	myfile.close();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	errCatch(cudaDeviceReset());

	return 0;
}


