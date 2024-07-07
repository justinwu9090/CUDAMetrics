#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>
#include <map>

// more standard libraries
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using namespace std;

// timekeeping
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;

// helper
#include "helper.cuh";

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}



__host__ map < string, string> addSequentialWrapper(int size, bool checkErrors = false)
{
	map < string, string> res;

	vector<int> a(size, 1);
	vector<int> b(size, 1);
	vector<int> c(size, 0);

	// Launch a kernel on the GPU with one thread for each element. use highres clock for wallclock time to complete
	auto t1 = high_resolution_clock::now();

	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
	/* Getting number of milliseconds as an integer. */
	auto t2 = high_resolution_clock::now();
	auto us_int = duration_cast<std::chrono::microseconds>(t2 - t1);

	// return map
	res["name"] = "sequential";
	res["array_size"] = to_string(size);
	res["duration_us"] = to_string((int)us_int.count());
	res["grid_dim"] = "";
	res["block_dim"] = "";
	return res;

}


// Helper function for using CUDA to add vectors in parallel.
__host__ map < string, string> addKernelWrapper(int size, bool checkErrors = false)
{
	map < string, string> res;

	vector<int> a(size, 1);
	vector<int> b(size, 1);
	vector<int> c(size, 0);

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	errCatch(cudaSetDevice(0));

	// Allocate GPU buffers for three vectors (two input, one output)    .
	errCatch(cudaMalloc((void**)&dev_c, size * sizeof(int)));
	errCatch(cudaMalloc((void**)&dev_a, size * sizeof(int)));
	errCatch(cudaMalloc((void**)&dev_b, size * sizeof(int)));

	// kernel dimensions (grid/block)
	dim3 gridDimInBlocks(ceil((int)(float)size / 32), 1, 1);
	dim3 blockDimInThreads(32, 1, 1);

	// Launch a kernel on the GPU with one thread for each element. use highres clock for wallclock time to complete
	auto t1 = high_resolution_clock::now();

	// Copy input vectors from host memory to GPU buffers.
	errCatch(cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice));
	errCatch(cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice));

	addKernel << <gridDimInBlocks, blockDimInThreads >> > (dev_c, dev_a, dev_b);
	// Check for any errors launching the kernel
	errCatch(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	errCatch(cudaDeviceSynchronize());

	/* Getting number of milliseconds as an integer. */
	auto t2 = high_resolution_clock::now();
	auto us_int = duration_cast<std::chrono::microseconds>(t2 - t1);

	// Copy output vector from GPU buffer to host memory.
	errCatch(cudaMemcpy(&c[0], dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	// error checking
	if (checkErrors)
	{
		for (int i = 0; i < c.size(); i++)
		{
			int ans = c[i];
			if (ans != a[0] + b[0])
			{
				printf("output wrong here\n");
			}
		}
	}

	errCatch(cudaFree(dev_c));
	errCatch(cudaFree(dev_a));
	errCatch(cudaFree(dev_b));

	// return map
	res["name"] = "CUDA naive";
	res["array_size"] = to_string(size);
	res["duration_us"] = to_string((int)us_int.count());
	res["grid_dim"] = to_string(gridDimInBlocks.x) + " " + to_string(gridDimInBlocks.y) + " " + to_string(gridDimInBlocks.z);
	res["block_dim"] = to_string(blockDimInThreads.x) + " " + to_string(blockDimInThreads.y) + " " + to_string(blockDimInThreads.z);
	//cout << "duration: " << us_int.count() << " us" << endl;
	cout << "duration: " << res["duration_us"] << " us" << endl;

	return res;
}

