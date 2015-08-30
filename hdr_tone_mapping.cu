#include "cuda_runtime.h"
#include "utils.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "device_atomic_functions.h"

const int blockSize = 1024;

__global__ 
void maxReduce
(
const float* const d_max_in,
float* d_max_out,
const int elements
)
{
	__shared__ float s_max[blockSize];

	int d_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// Copy values to shared memory for threads that are in the boundary of the image
	if (d_1D_pos < elements)
	{
		s_max[tid] = d_max_in[d_1D_pos];
	}
	else
	{
		s_max[tid] = std::numeric_limits<float>::min(); // Apply a min float value to out of bounds values so that it doesn't affect other threads.
	}

	__syncthreads();

	for (int i = blockSize / 2; i > 0; i >>= 1)
	{
		if (tid < i)
		{
			s_max[tid] = fmax(s_max[tid], s_max[tid + i]);
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		d_max_out[blockIdx.x] = s_max[0];
	}
}

__global__ void minReduce
(
const float* const d_min_in,
float* d_min_out,
const int elements
)
{
	__shared__ float s_min[blockSize];

	int d_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// Copy values to shared memory for threads that are in the boundary of the image
	if (d_1D_pos < elements)
	{
		s_min[tid] = d_min_in[d_1D_pos];
	}
	else
	{
		s_min[tid] = std::numeric_limits<float>::max(); // Apply a min float value to out of bounds values so that it doesn't affect other threads.
	}

	__syncthreads();

	for (int i = blockSize / 2; i > 0; i >>= 1)
	{
		if (tid < i)
		{
			s_min[tid] = fmin(s_min[tid], s_min[tid + i]);
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		d_min_out[blockIdx.x] = s_min[0];
	}
}

__global__
void createHistogram
(
const float* const d_logLuminance,
int* histogram,
const float lumRange,
const float min_logLum,
const int elements,
const size_t numBins
)
{
	int d_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;

	// Return if the thread is out of bounds
	if (d_1D_pos > elements) return;


	// Get luminance value for this thread's position
	int threadLuminance = d_logLuminance[d_1D_pos];

	// Classify the value in the histogram
	int bin = (threadLuminance - min_logLum) / lumRange * numBins;
	
	// Atomically add the value in the histogram. Add a 1, as bin holds the total value of the threads in the bin.
	// Could be implemented without atomics (using local histograms). Local histograms would then be reduced into a global histogram.
	// The more bins you're using, the slower the kernel using atomic operations will be. 
	atomicAdd(&histogram[bin], 1);
}



// Minimal version of a work-efficient exclusive scan. 
// Doesn't implement offset shared memory bank conflict. The array would have to load its two elements into the temp[] array from different separate halves of the histogram. 
// Doesn't allow for arrays of arbitrary size.
__global__
void exclusiveScan
(unsigned int* d_cdf, 
int const* histrogram, 
const size_t numBins
)
{
	extern __shared__ float temp[];

	int tid = threadIdx.x;
	
	int pout = 0, pin = 1;

	temp[pout * numBins + tid] = (tid > 0) ? histrogram[tid - 1] : 0;
	__syncthreads();

	// Offset for an exclusive scan
	for (int offset = 1; offset < numBins; offset *= 2)
	{
		pout = 1 - pout;
		pin = 1 - pout;
		if (tid >= offset)
			temp[pout*numBins + tid] += temp[pin * numBins + tid - offset];
		else
			temp[pout*numBins + tid] = temp[pin * numBins + tid];
		__syncthreads();
	}
	d_cdf[tid] = temp[pout * numBins + tid];
}


void your_histogram_and_prefixsum(const float* const h_logLuminance,
	unsigned int* const h_cumulativeDistribution,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	int **d_cumulativeDistribution;
	float **d_logLuminance;
	float *d_array;
	float **d_luminanceOut = &d_array;
	float **d_temp;
	int elements;
	int gridSize;

	// Allocate a device version of h_logLuminance and copy the contents from host to device
	checkCudaErrors(cudaMalloc(&d_logLuminance, sizeof(float) * numRows * numCols));
	checkCudaErrors(cudaMemcpy(&d_logLuminance, h_logLuminance, sizeof(float)* numRows * numCols, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_array, sizeof(float) * numRows * numCols));


	// Compute the max_logLum in the d_logLuminance channel using a reduce primitive implementation
	gridSize = numRows * numCols;
	do
	{
		elements = gridSize;
		gridSize = ceil((float)elements / (float)blockSize);

		maxReduce << < gridSize, blockSize >> >(*d_logLuminance, *d_out, elements);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		if (gridSize == 1)
		{
			checkCudaErrors(cudaMemcpy(&max_logLum, d_luminanceOut[0], sizeof(float), cudaMemcpyDeviceToHost));
		}

		//exchange input array and output array
		d_temp = d_logLuminance;
		d_logLuminance = d_luminanceOut;
		d_out = d_temp;

	} while (gridSize > 1);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Compute the max_logLum in the d_logLuminance channel using a reduce primitive implementation
	// 'Reset' d_logLuminance as it was written to in maxReduce
	checkCudaErrors(cudaMemcpy(&d_logLuminance, h_logLuminance, sizeof(float)* numRows * numCols, cudaMemcpyHostToDevice));
	gridSize = numRows * numCols;
	do
	{
		elements = gridSize;
		gridSize = ceil((float)elements / (float)blockSize);

		minReduce << < gridSize, blockSize >> >(*d_logLuminance, *d_out, elements);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		if (gridSize == 1)
		{
			checkCudaErrors(cudaMemcpy(&min_logLum, *d_luminanceOut, sizeof(float), cudaMemcpyDeviceToHost));
		}

		//exchange input array and output array
		d_temp = d_logLuminance;
		d_logLuminance = d_luminanceOut;
		d_luminanceOut = d_temp;

	} while (gridSize > 1);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	// Calculate the luminance range
	int lumRange = max_logLum - min_logLum;


	// Create a histogram from the values in the logLuminance channel
	// Formula:
	// bin = (lum[i] - lumMin) / lumRange * numBins * 3) 

	// Create a and allocate the histogram
	int *histogram;
	checkCudaErrors(cudaMalloc(&histogram, sizeof(int) * numBins));
	// Initialize histogram values to 0
	checkCudaErrors(cudaMemset(histogram, 0, numBins));

	elements = numRows * numCols;
	gridSize = ceil((float)elements / (float)blockSize);


	createHistogram << < gridSize, blockSize >> > (d_logLuminance, histogram, lumRange, min_logLum, elements, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Allocate a device version of h_cumulativeDistribution and copy the contents from host to device
	checkCudaErrors(cudaMalloc(&d_cumulativeDistribution, sizeof(float) * numRows * numCols));
	checkCudaErrors(cudaMemcpy(&d_cumulativeDistribution, h_cumulativeDistribution, sizeof(unsigned int) * numBins, cudaMemcpyHostToDevice));

	exclusiveScan << <1, blockSize, sizeof(int) * blockSize * 2 >> > (d_cumulativeDistribution, histogram, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_logLuminance));
	checkCudaErrors(cudaFree(d_cumulativeDistribution));
	checkCudaErrors(cudaFree(d_array));
	checkCudaErrors(cudaFree(histogram));
}
