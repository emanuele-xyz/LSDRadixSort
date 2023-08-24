#include <iostream>
#include <stdint.h>

// Windows only
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <intrin.h>

#include "Utils.h"
#include "CudaUtils.h"

/*
* Extract the i-th group of r bits from n
*/
#define GET_R_BITS(n, r, i) (((1 << r) - 1) & (n >> (i * r)))

/*
* in:    input array (will be modified)
* out:   output array
* count: number of elements to sort
* r:     number of bits to consider as keys (any factor of 32 apart from itself)
*/
void LSDRadixSort(uint32_t* in, uint32_t* out, int count, int* histogram, int r)
{
	int iterations = (sizeof(*in) * 8) / r;
	for (int i = 0; i < iterations; i++)
	{
		memset(histogram, 0, sizeof(*histogram) * (1 << r));

		// build histogram
		for (int j = 0; j < count; j++)
		{
			uint32_t val = in[j];
			int key = GET_R_BITS(val, r, i);
			histogram[key]++;
		}

		// prefix sum of histogram
		for (int j = 1; j < (1 << r); j++)
		{
			histogram[j] += histogram[j - 1];
		}

		// permute input elements into output
		for (int j = count - 1; j >= 0; j--)
		{
			uint32_t val = in[j];
			int key = GET_R_BITS(val, r, i);
			histogram[key]--;
			out[histogram[key]] = val;
		}

		// copy output array into input array
		memcpy(in, out, sizeof(*in) * count);
	}
}

void PrefixSum(uint32_t* a, int count)
{
	for (int i = 1; i < count; i++)
	{
		a[i] += a[i - 1];
	}
	for (int i = count - 1; i >= 1; i--)
	{
		a[i] = a[i - 1];
	}
	a[0] = 0;
}

__device__ void SMEMUpSweep(uint32_t* smem, int bdim, int tid)
{
	for (int d = 0; (1 << d) < bdim; d++)
	{
		int offset = (1 << (d + 1));
		int bias = offset - 1;
		int shift = (1 << d);

		if (tid < (bdim >> (d + 1)))
		{
			int index = bias + tid * offset;
			int left = index - shift;
			smem[index] += smem[left];
		}
		__syncthreads();
	}
}

__device__ void SMEMDownSweep(uint32_t* smem, int bdim, int tid)
{
	for (int d = 0; (1 << d) < bdim; d++)
	{
		int offset = (bdim >> d);
		int bias = offset - 1;
		int shift = (bdim >> (d + 1));

		if (tid < (1 << d))
		{
			int index = bias + tid * offset;
			int left = index - shift;
			int l = smem[index];
			int r = smem[index] + smem[left];
			smem[left] = l;
			smem[index] = r;
		}
		__syncthreads();
	}
}

__global__ void BlockPrefixSumKernel(uint32_t* a, uint32_t* block_sums)
{
	extern __shared__ uint32_t smem[];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int bdim = blockDim.x;
	int i = bid * bdim + tid;

	// load array into smem
	smem[tid] = a[i];
	__syncthreads();

	SMEMUpSweep(smem, bdim, tid);
	if (tid == 0)
	{
		block_sums[bid] = smem[bdim - 1];
		smem[bdim - 1] = 0;
	}
	__syncthreads();
	SMEMDownSweep(smem, bdim, tid);

	// write smem into array
	a[i] = smem[tid];
}

#define PRINT_TIMINGS

#define SEQ_RADIX_SORT_TEST_ELEMS_COUNT 16
#define SEQ_RADIX_SORT_TEST_ELEMS_MIN 0
#define SEQ_RADIX_SORT_TEST_ELEMS_MAX 10
#define SEQ_RADIX_SORT_TEST_ELEMS_R 4

void TestSequentialLSDRadixSort()
{
	#ifdef PRINT_TIMINGS
	std::cout << "-- Test sequential LSD radix sort --" << std::endl;
	#endif

	RNG rng = RNG(0, SEQ_RADIX_SORT_TEST_ELEMS_MIN, SEQ_RADIX_SORT_TEST_ELEMS_MAX);

	int count = SEQ_RADIX_SORT_TEST_ELEMS_COUNT;
	uint32_t* a = (uint32_t*)(calloc(count, sizeof(*a)));
	uint32_t* b = (uint32_t*)(calloc(count, sizeof(*b)));
	uint32_t* c = (uint32_t*)(calloc(count, sizeof(*c)));
	int* h = (int*)(calloc(1 << SEQ_RADIX_SORT_TEST_ELEMS_R, sizeof(*h)));

	// populate a and c
	for (int i = 0; i < count; i++)
	{
		int elem = rng.Get();
		a[i] = elem;
		c[i] = elem;
	}

	// sort a writing result in b using LSD radix sort
	float lsd_radix_sort_ms = 0;
	{
		int64_t start = GetTimestamp();
		LSDRadixSort(a, b, count, h, SEQ_RADIX_SORT_TEST_ELEMS_R);
		int64_t end = GetTimestamp();
		lsd_radix_sort_ms = GetElapsedMS(start, end);
	}

	// sort c using standard library sort
	float std_sort_ms = 0;
	{
		int64_t start = GetTimestamp();
		std::sort(c, c + count);
		int64_t end = GetTimestamp();
		std_sort_ms = GetElapsedMS(start, end);
	}

	#ifdef PRINT_ARRAY
	PrintArray('b', b, count);
	PrintArray('c', c, count);
	#endif

	#ifdef PRINT_TIMINGS
	std::cout << "Sequential LSD Radix Sort: " << lsd_radix_sort_ms << " ms" << std::endl;
	std::cout << "Sequential STD Sort: " << std_sort_ms << " ms" << std::endl;
	std::cout << "Speedup: x" << std_sort_ms / lsd_radix_sort_ms << std::endl;
	#endif

	CheckArrays(b, c, count);

	free(h);
	free(c);
	free(b);
	free(a);
}

#define BLOCK_PREFIX_SUM_TEST_ELEMS_COUNT 1024
#define BLOCK_PREFIX_SUM_TEST_ELEMS_MIN 0
#define BLOCK_PREFIX_SUM_TEST_ELEMS_MAX 10

void TestBlockPrefixSumKernel()
{
	#ifdef PRINT_TIMINGS
	std::cout << "-- Test parallel exclusive prefix sum --" << std::endl;
	#endif

	RNG rng = RNG(0, BLOCK_PREFIX_SUM_TEST_ELEMS_MIN, BLOCK_PREFIX_SUM_TEST_ELEMS_MAX);

	int count = BLOCK_PREFIX_SUM_TEST_ELEMS_COUNT;
	size_t size = count * sizeof(uint32_t);
	size_t blocks_size = 1 * sizeof(uint32_t);
	uint32_t* h_a = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_b = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* d_a = (uint32_t*)MyCudaMalloc(size);
	uint32_t* d_block_sums = (uint32_t*)MyCudaMalloc(blocks_size);

	// populate a
	for (int i = 0; i < count; i++)
	{
		int elem = rng.Get();
		h_a[i] = elem;
	}

	// paralle prefix sum
	float parallel_ms = 0;
	{
		cudaEvent_t start = MyCudaEventCreate();
		cudaEvent_t end = MyCudaEventCreate();
		CUDA_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaEventRecord(start));
		BlockPrefixSumKernel << <1, count, size >> > (d_a, d_block_sums);
		CUDA_CALL(cudaEventRecord(end));
		CUDA_CALL(cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost));
		parallel_ms = MyCudaEventElapsedTime(start, end);
	}

	float sequential_ms = 0;
	{
		int64_t start = GetTimestamp();
		PrefixSum(h_a, count);
		int64_t end = GetTimestamp();
		sequential_ms = GetElapsedMS(start, end);
	}

	#ifdef PRINT_ARRAY
	PrintArray('a', h_a, count);
	PrintArray('b', h_b, count);
	#endif

	#ifdef PRINT_TIMINGS
	std::cout << "Prefix Sum Sequential: " << sequential_ms << " ms" << std::endl;
	std::cout << "Prefix Sum Parallel: " << parallel_ms << " ms" << std::endl;
	std::cout << "Speedup: x" << sequential_ms / parallel_ms << std::endl;
	#endif

	CheckArrays(h_a, h_b, count);

	CUDA_CALL(cudaFree(d_block_sums));
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFreeHost(h_b));
	CUDA_CALL(cudaFreeHost(h_a));
}

int main()
{
	CheckForHostLeaks();

	TestSequentialLSDRadixSort();
	TestBlockPrefixSumKernel();

	return 0;
}
