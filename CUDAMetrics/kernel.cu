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
using namespace std;

// timekeeping
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;

__host__ void errCatch(cudaError_t);
__host__ void addWithCuda(vector<int>& c, vector<int>& a, vector<int>& b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

cudaDeviceProp dumpDeviceProperties()
{
	int deviceID;
	cudaDeviceProp props;

	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&props, deviceID);
	return props;
}

void simpleAdd(int arraySize)
{
	ofstream myfile;
	myfile.open("test.csv");
	vector<int> a(arraySize, 1);
	vector<int> b(arraySize, 1);
	vector<int> c(arraySize, 0);
	addWithCuda(c, a, b, arraySize);
}
int main()
{
	cudaDeviceProp props = dumpDeviceProperties();
	cout << "GPU: " << props.name << endl;
	cout << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor << endl;
	cout << "maxBlocksPerMultiProcessor: " << props.maxBlocksPerMultiProcessor << endl;
	cout << "multiProcessorCount: " << props.multiProcessorCount << endl;

	int maxoccupancythreads = props.multiProcessorCount * props.maxBlocksPerMultiProcessor * props.maxThreadsPerMultiProcessor;
	cout << "maxoccupancythreads: " << maxoccupancythreads << endl;


	const unsigned int arraySize = maxoccupancythreads;
	

	// Add vectors in parallel.
	simpleAdd(arraySize);
	simpleAdd(arraySize*2);
	simpleAdd(arraySize*3);
	
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	errCatch(cudaDeviceReset());

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
__host__ void addWithCuda(vector<int>& c, vector<int>& a, vector<int>& b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.

	errCatch(cudaSetDevice(0));

	// Allocate GPU buffers for three vectors (two input, one output)    .
	errCatch(cudaMalloc((void**)&dev_c, size * sizeof(int)));
	errCatch(cudaMalloc((void**)&dev_a, size * sizeof(int)));
	errCatch(cudaMalloc((void**)&dev_b, size * sizeof(int)));

	const int ROUNDS = 5;
	for (int i = 0; i < ROUNDS; i++)
	{

		// Copy input vectors from host memory to GPU buffers.
		errCatch(cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice));
		errCatch(cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice));

		auto t1 = high_resolution_clock::now();
		// Launch a kernel on the GPU with one thread for each element.
		dim3 gridDimInBlocks(ceil((float)size / 32), 1, 1);
		dim3 blockDimInThreads(32, 1, 1);
		addKernel << <gridDimInBlocks, blockDimInThreads >> > (dev_c, dev_a, dev_b);
		auto t2 = high_resolution_clock::now();

		/* Getting number of milliseconds as an integer. */
		auto us_int = duration_cast<std::chrono::microseconds>(t2 - t1);

		// Check for any errors launching the kernel
		errCatch(cudaGetLastError());

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		errCatch(cudaDeviceSynchronize());

		// Copy output vector from GPU buffer to host memory.
		errCatch(cudaMemcpy(&c[0], dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

		for (int i = 0; i < c.size(); i++)
		{
			int ans = c[i];
			if (ans != a[0] + b[0])
			{
				printf("output wrong here\n");
			}
		}

		printf("%d us\n", (int) us_int.count());
	}

	//Error:
	errCatch(cudaFree(dev_c));
	errCatch(cudaFree(dev_a));
	errCatch(cudaFree(dev_b));
}

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
