#include <iostream>
#include <stdint.h>

/*
	TODO: watch this https://www.youtube.com/watch?v=fsC3QeZHM1U
*/

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
void LSDRadixSort(uint32_t* in, uint32_t* out, int count, uint32_t* histogram, int r)
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
#define SEQ_RADIX_SORT_TEST_ELEMS_MAX UINT32_MAX
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
	uint32_t* h = (uint32_t*)(calloc(1 << SEQ_RADIX_SORT_TEST_ELEMS_R, sizeof(*h)));

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
	std::cout << "-- Test block exclusive prefix sum --" << std::endl;
	#endif

	RNG rng = RNG(0, BLOCK_PREFIX_SUM_TEST_ELEMS_MIN, BLOCK_PREFIX_SUM_TEST_ELEMS_MAX);

	int count = BLOCK_PREFIX_SUM_TEST_ELEMS_COUNT;
	size_t size = count * sizeof(uint32_t);
	size_t blocks_size = 1 * sizeof(uint32_t);
	uint32_t* h_a = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_b = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* d_a = (uint32_t*)MyCudaMalloc(size);
	uint32_t* d_block_sums = (uint32_t*)MyCudaMalloc(blocks_size);

	for (int i = 0; i < count; i++) h_a[i] = rng.Get();

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
		CUDA_CALL(cudaEventDestroy(end));
		CUDA_CALL(cudaEventDestroy(start));
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
	std::cout << "Prefix Sum Block: " << parallel_ms << " ms" << std::endl;
	std::cout << "Speedup: x" << sequential_ms / parallel_ms << std::endl;
	#endif

	CheckArrays(h_a, h_b, count);

	CUDA_CALL(cudaFree(d_block_sums));
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFreeHost(h_b));
	CUDA_CALL(cudaFreeHost(h_a));
}

int GetGPUPrefixSumBlockSumsCount(int count, int threads_per_block)
{
	MYASSERT(count % threads_per_block == 0);
	int total_block_sums_count = 0;
	while (count > threads_per_block)
	{
		MYASSERT(count % threads_per_block == 0);
		int block_sums_count = count / threads_per_block;
		total_block_sums_count += block_sums_count;
		count = block_sums_count;
	}
	return total_block_sums_count + 1;
}

__global__ void AddBlockSumsKernel(uint32_t* a, uint32_t* block_sums)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int i = blockDim.x * bid + tid;
	a[i] += block_sums[bid];
}

void GPUPrefixSum(uint32_t* d_a, int count, int threads_per_block, uint32_t* d_block_sums)
{
	if (count <= threads_per_block)
	{
		int smem = count * sizeof(uint32_t);
		BlockPrefixSumKernel << <1, count, smem >> > (d_a, d_block_sums);
	}
	else
	{
		int smem = threads_per_block * sizeof(uint32_t);
		int blocks = count / threads_per_block;
		BlockPrefixSumKernel << <blocks, threads_per_block, smem >> > (d_a, d_block_sums);
		GPUPrefixSum(d_block_sums, blocks, threads_per_block, &d_block_sums[blocks]);
		// Skip the first block, since the exclusive block sum will have 0
		AddBlockSumsKernel << <blocks - 1, threads_per_block >> > (&d_a[threads_per_block], &d_block_sums[1]);
	}
}

void TestGPUPrefixSum(int count, int threads_per_block, int min, int max)
{
	#ifdef PRINT_TIMINGS
	std::cout << "-- Test exclusive prefix sum --" << std::endl;
	#endif

	RNG rng = RNG(0, min, max);

	// allocate
	int block_sums_count = GetGPUPrefixSumBlockSumsCount(count, threads_per_block);
	size_t size = count * sizeof(uint32_t);
	size_t block_sums_size = block_sums_count * sizeof(uint32_t);
	uint32_t* h_a = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_b = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* d_a = (uint32_t*)MyCudaMalloc(size);
	uint32_t* d_block_sums = (uint32_t*)MyCudaMalloc(block_sums_size);

	// populate input
	for (int i = 0; i < count; i++) h_a[i] = rng.Get();

	// parallel prefix sum
	float parallel_ms = 0;
	{
		cudaEvent_t start = MyCudaEventCreate();
		cudaEvent_t end = MyCudaEventCreate();
		CUDA_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaEventRecord(start));
		GPUPrefixSum(d_a, count, threads_per_block, d_block_sums);
		CUDA_CALL(cudaEventRecord(end));
		CUDA_CALL(cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost));
		parallel_ms = MyCudaEventElapsedTime(start, end);
		CUDA_CALL(cudaEventDestroy(end));
		CUDA_CALL(cudaEventDestroy(start));
	}

	// sequential prefix sum
	float sequential_ms = 0;
	{
		int64_t start = GetTimestamp();
		PrefixSum(h_a, count);
		int64_t end = GetTimestamp();
		sequential_ms = GetElapsedMS(start, end);
	}

	// print arrays
	#ifdef PRINT_ARRAY
	PrintArray('a', h_a, count);
	PrintArray('b', h_b, count);
	#endif

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "Prefix sum of " << (double)(size) / (1024.0 * 1024.0 * 1024.0) << " GB of data" << std::endl;
	std::cout << "Threads per block: " << threads_per_block << std::endl;
	std::cout << "Prefix Sum Sequential: " << sequential_ms << " ms" << std::endl;
	std::cout << "GPU Prefix Sum: " << parallel_ms << " ms" << std::endl;
	std::cout << "Speedup: x" << sequential_ms / parallel_ms << std::endl;
	#endif

	// check arrays
	CheckArrays(h_a, h_b, count);

	// deallocate
	CUDA_CALL(cudaFree(d_block_sums));
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFreeHost(h_b));
	CUDA_CALL(cudaFreeHost(h_a));
}

__device__ void SMEMLSDBinaryRadixSort(uint32_t* a, int tid, int bdim)
{
	// 32 passes for 32 bits numbers
	for (int i = 0; i < 32; i++)
	{
		uint32_t val = a[tid];
		uint32_t bit = GET_R_BITS(val, 1, i);
		// invert bit and write it to a
		a[tid] = bit ? 0 : 1;
		__syncthreads();

		// prefix sum of inverted bits
		SMEMUpSweep(a, bdim, tid);
		uint32_t total_falses = a[bdim - 1];
		if (tid == 0) a[bdim - 1] = 0;
		__syncthreads();
		SMEMDownSweep(a, bdim, tid);

		// now a holds the destination for false keys
		// destination for true key
		uint32_t t = (uint32_t)tid - a[tid] + total_falses;
		// destination for val
		uint32_t d = bit ? t : a[tid];
		a[d] = val;
	}
	__syncthreads();
}

__global__ void LSDBinaryRadixSortKernel(uint32_t* a)
{
	extern __shared__ uint32_t smem[];

	int tid = threadIdx.x;

	// load array in smem
	smem[tid] = a[tid];
	__syncthreads();

	SMEMLSDBinaryRadixSort(smem, tid, blockDim.x);

	// write smem into array
	a[tid] = smem[tid];
}

#define LSD_BINARY_RADIX_SORT_TEST_ELEMS_COUNT (1024)
#define LSD_BINARY_RADIX_SORT_TEST_ELEMS_MIN 0
#define LSD_BINARY_RADIX_SORT_TEST_ELEMS_MAX UINT32_MAX

void TestLSDBinaryRadixSort()
{
	#ifdef PRINT_TIMINGS
	std::cout << "-- Test LSD Binary Radix Sort --" << std::endl;
	#endif

	RNG rng = RNG(0, LSD_BINARY_RADIX_SORT_TEST_ELEMS_MIN, LSD_BINARY_RADIX_SORT_TEST_ELEMS_MAX);

	// allocate
	int count = LSD_BINARY_RADIX_SORT_TEST_ELEMS_COUNT;
	size_t size = count * sizeof(uint32_t);
	uint32_t* h_a = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_b = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* d_a = (uint32_t*)MyCudaMalloc(size);

	// populate input
	for (int i = 0; i < count; i++) h_a[i] = rng.Get();

	// parallel lsd binary radix sort
	float parallel_lsd_binary_radix_sort_ms = 0;
	{
		cudaEvent_t start = MyCudaEventCreate();
		cudaEvent_t end = MyCudaEventCreate();
		CUDA_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaEventRecord(start));
		LSDBinaryRadixSortKernel << <1, count, size >> > (d_a);
		CUDA_CALL(cudaEventRecord(end));
		CUDA_CALL(cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost));
		parallel_lsd_binary_radix_sort_ms = MyCudaEventElapsedTime(start, end);
		CUDA_CALL(cudaEventDestroy(end));
		CUDA_CALL(cudaEventDestroy(start));
	}

	// sequential sort
	float std_sort_ms = 0;
	{
		int64_t start = GetTimestamp();
		std::sort(h_a, h_a + count);
		int64_t end = GetTimestamp();
		std_sort_ms = GetElapsedMS(start, end);
	}

	// print arrays
	#ifdef PRINT_ARRAY
	PrintArray('a', h_a, count);
	PrintArray('b', h_b, count);
	#endif

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "STD Sort: " << std_sort_ms << " ms" << std::endl;
	std::cout << "Parallel LSD Binary Radix Sort: " << parallel_lsd_binary_radix_sort_ms << " ms" << std::endl;
	#endif

	// check arrays
	CheckArrays(h_a, h_b, count);

	// deallocate
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFreeHost(h_b));
	CUDA_CALL(cudaFreeHost(h_a));
}

/*
	a: m x n matrix
	b: n x m matrix
*/
void Transpose(uint32_t* a, uint32_t* b, int m, int n)
{
	for (int row = 0; row < m; row++)
	{
		for (int col = 0; col < n; col++)
		{
			int a_i = row * n + col;
			int b_i = col * m + row;
			b[b_i] = a[a_i];
		}
	}
}

/*
	a : m x n matrix
	b : n x m matrix
*/
__global__ void TransposeNaiveKernel(uint32_t* a, uint32_t* b, int m, int n)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= m || j >= n) return;
	b[j * m + i] = a[i * n + j];
}

/*
	a : m x n matrix
	b : n x m matrix
*/
__global__ void TransposeSMEMKernel(uint32_t* a, uint32_t* b, int m, int n)
{
	extern __shared__ uint32_t smem[];

	int t_i = threadIdx.y;
	int t_j = threadIdx.x;
	int t_cols = blockDim.x;
	int t_rows = blockDim.y;
	int a_i = blockIdx.y * blockDim.y + t_i;
	int a_j = blockIdx.x * blockDim.x + t_j;
	int b_i = blockIdx.x * blockDim.x + t_i;
	int b_j = blockIdx.y * blockDim.y + t_j;

	// copy matrix block into smem
	if (a_i < m && a_j < n)
	{
		smem[t_i * t_cols + t_j] = a[a_i * n + a_j];
	}
	__syncthreads();

	// write transposed smem block
	if (b_i < n && b_j < m)
	{
		b[b_i * m + b_j] = smem[t_j * t_rows + t_i];
	}
}

#define TRANSPOSE_TEST_M (1024)
#define TRANSPOSE_TEST_N (1024)
#define TRANSPOSE_TEST_BLOCK_DIM 32
#define TRANSPOSE_TEST_MIN_ELEM 0
#define TRANSPOSE_TEST_MAX_ELEM 9

void TestTranspose()
{
	#ifdef PRINT_TIMINGS
	std::cout << "-- Test Transpose --" << std::endl;
	#endif

	RNG rng = RNG(0, TRANSPOSE_TEST_MIN_ELEM, TRANSPOSE_TEST_MAX_ELEM);

	// allocate
	int m = TRANSPOSE_TEST_M;
	int n = TRANSPOSE_TEST_N;
	int count = m * n;
	int block_dim = TRANSPOSE_TEST_BLOCK_DIM;
	size_t size = count * sizeof(uint32_t);
	uint32_t* h_a = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_b = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_c = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* d_a = (uint32_t*)MyCudaMalloc(size);
	uint32_t* d_b = (uint32_t*)MyCudaMalloc(size);

	#ifdef PRINT_TIMINGS
	std::cout << "Transpose of " << (double)(size) / (1024.0 * 1024.0 * 1024.0) << " GB of data" << std::endl;
	#endif

	// populate input
	for (int i = 0; i < count; i++) h_a[i] = rng.Get();

	// sequential transpose
	float sequential_ms = 0.0f;
	{
		int64_t start = GetTimestamp();
		Transpose(h_a, h_b, m, n);
		int64_t end = GetTimestamp();
		sequential_ms = GetElapsedMS(start, end);
	}

	float gpu_naive_ms = 0.0f;
	{
		cudaEvent_t start = MyCudaEventCreate();
		cudaEvent_t end = MyCudaEventCreate();
		CUDA_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaEventRecord(start));
		dim3 block(block_dim, block_dim);
		dim3 grid((n + block_dim - 1) / block_dim, (m + block_dim - 1) / block_dim);
		TransposeNaiveKernel << <grid, block >> > (d_a, d_b, m, n);
		CUDA_CALL(cudaEventRecord(end));
		CUDA_CALL(cudaMemcpy(h_c, d_b, size, cudaMemcpyDeviceToHost));
		gpu_naive_ms = MyCudaEventElapsedTime(start, end);
		CUDA_CALL(cudaEventDestroy(end));
		CUDA_CALL(cudaEventDestroy(start));
	}

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "Sqeuential Transpose: " << sequential_ms << " ms" << std::endl;
	std::cout << "GPU Naive Transpose: " << gpu_naive_ms << " ms - Speedup: x" << sequential_ms / gpu_naive_ms << std::endl;
	#endif

	// check matrices
	CheckArrays(h_b, h_c, count);

	float gpu_smem_ms = 0.0f;
	{
		cudaEvent_t start = MyCudaEventCreate();
		cudaEvent_t end = MyCudaEventCreate();
		CUDA_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaEventRecord(start));
		dim3 block(block_dim, block_dim);
		dim3 grid((n + block_dim - 1) / block_dim, (m + block_dim - 1) / block_dim);
		size_t smem = block_dim * block_dim * sizeof(uint32_t);
		TransposeSMEMKernel << <grid, block, smem >> > (d_a, d_b, m, n);
		CUDA_CALL(cudaEventRecord(end));
		CUDA_CALL(cudaMemcpy(h_c, d_b, size, cudaMemcpyDeviceToHost));
		gpu_smem_ms = MyCudaEventElapsedTime(start, end);
		CUDA_CALL(cudaEventDestroy(end));
		CUDA_CALL(cudaEventDestroy(start));
	}

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "GPU SMEM Transpose: " << gpu_smem_ms << " ms - Speedup: x" << sequential_ms / gpu_smem_ms << std::endl;
	#endif

	// check matrices
	CheckArrays(h_b, h_c, count);

	// deallocate
	cudaFree(d_b);
	cudaFree(d_a);
	cudaFreeHost(h_c);
	cudaFreeHost(h_b);
	cudaFreeHost(h_a);
}

/*
	bdim: 64   r: 1  lct: 2     ratio: 0.03125
	bdim: 64   r: 2  lct: 4     ratio: 0.0625
	bdim: 64   r: 4  lct: 16    ratio: 0.25
	bdim: 64   r: 8  lct: 256   ratio: 4
	bdim: 64   r: 16 lct: 65536 ratio: 1024

	bdim: 128  r: 1  lct: 2	    ratio: 0.015625
	bdim: 128  r: 2  lct: 4	    ratio: 0.03125
	bdim: 128  r: 4  lct: 16    ratio: 0.125
	bdim: 128  r: 8  lct: 256   ratio: 2
	bdim: 128  r: 16 lct: 65536 ratio: 512

	bdim: 256  r: 1  lct: 2	    ratio: 0.0078125
	bdim: 256  r: 2  lct: 4	    ratio: 0.015625
	bdim: 256  r: 4  lct: 16    ratio: 0.0625
	bdim: 256  r: 8  lct: 256   ratio: 1
	bdim: 256  r: 16 lct: 65536 ratio: 256

	bdim: 512  r: 1  lct: 2     ratio: 0.00390625
	bdim: 512  r: 2  lct: 4     ratio: 0.0078125
	bdim: 512  r: 4  lct: 16    ratio: 0.03125
	bdim: 512  r: 8  lct: 256   ratio: 0.5
	bdim: 512  r: 16 lct: 65536 ratio: 128

	bdim: 1024 r: 1  lct: 2     ratio: 001953125
	bdim: 1024 r: 2  lct: 4     ratio: 0.00390625
	bdim: 1024 r: 4  lct: 16    ratio: 0.015625
	bdim: 1024 r: 8  lct: 256   ratio: 0.25
	bdim: 1024 r: 16 lct: 65536 ratio: 64
*/

/*
	grid:  how many histograms
	block: how many array elements per histogram
*/
void BuildHistogramsCPU(uint32_t* a, uint32_t* h, int count, int r, int bit_group, int grid, int block)
{
	int h_count = (1 << r);

	for (int g = 0; g < grid; g++)
	{
		for (int b = 0; b < block; b++)
		{
			int a_i = g * block + b;
			uint32_t val = a[a_i];
			int key = GET_R_BITS(val, r, bit_group);
			int h_i = g * h_count + key;
			h[h_i] += 1;
		}
	}
}

__global__ void BuildHistogramsKernel(uint32_t* a, uint32_t* h, int count, int r, int bit_group)
{
	extern __shared__ uint32_t smem[];

	int bdim = blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int idx = bid * bdim + tid;

	// zero initialize smem
	int lct_count = 1 << r;
	float cells_per_thread_ratio = (float)(lct_count) / (float)(bdim);
	int cells_per_thread = cells_per_thread_ratio < 1.0f ? 1 : (int)(cells_per_thread_ratio + 0.5f);
	for (int i = 0; i < cells_per_thread; i++)
	{
		int smem_i = tid * cells_per_thread + i;
		if (smem_i < lct_count)
		{
			smem[smem_i] = 0;
		}
	}
	__syncthreads();

	// build histogram in smem
	if (idx < count)
	{
		uint32_t val = a[idx];
		int key = GET_R_BITS(val, r, bit_group);
		atomicInc(&smem[key], UINT32_MAX);
	}
	__syncthreads();

	// write histogram to global memory
	for (int i = 0; i < cells_per_thread; i++)
	{
		int smem_i = tid * cells_per_thread + i;
		int h_i = bid * lct_count + smem_i;
		if (smem_i < lct_count)
		{
			h[h_i] = smem[smem_i];
		}
	}
}

void TestBuildHistogram(int count, int block, int r, int bit_group, int min, int max)
{
	#ifdef PRINT_TIMINGS
	std::cout << "-- Test Build Histogram --" << std::endl;
	#endif

	int grid = (count + block - 1) / block;
	size_t h_count = (1 << r);
	size_t h_total_count = h_count * grid;

	// get sizes
	size_t size = count * sizeof(uint32_t);
	size_t h_total_size = h_total_count * sizeof(uint32_t);

	#ifdef PRINT_TIMINGS
	std::cout << "Elements: " << (double)(size) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
	std::cout << "Histograms: " << (double)(h_total_size) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
	std::cout << "Block Size: " << block << std::endl;
	std::cout << "R: " << r << std::endl;
	std::cout << "Bit Group: " << bit_group << std::endl;
	#endif

	if (h_total_size > size)
	{
		#ifdef PRINT_TIMINGS
		std::cout << "SKIP: histogram is bigger than input" << std::endl;
		#endif
		return;
	}

	// allocate
	uint32_t* h_a = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_h1 = (uint32_t*)MyCudaHostAlloc(h_total_size);
	uint32_t* h_h2 = (uint32_t*)MyCudaHostAlloc(h_total_size);
	uint32_t* d_a = (uint32_t*)MyCudaMalloc(size);
	uint32_t* d_h = (uint32_t*)MyCudaMalloc(h_total_size);

	// populate input
	RNG rng = RNG(0, min, max);
	for (int i = 0; i < count; i++) h_a[i] = rng.Get();

	// cpu implementation
	float cpu_ms = 0.0f;
	{
		int64_t start = GetTimestamp();
		BuildHistogramsCPU(h_a, h_h1, count, r, bit_group, grid, block);
		int64_t end = GetTimestamp();
		cpu_ms = GetElapsedMS(start, end);
	}

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "CPU " << cpu_ms << " ms" << std::endl;
	#endif

	// gpu implementation
	float gpu_ms = 0.0f;
	{
		cudaEvent_t start = MyCudaEventCreate();
		cudaEvent_t end = MyCudaEventCreate();
		CUDA_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaEventRecord(start));
		size_t smem = h_count * sizeof(uint32_t);
		BuildHistogramsKernel << <grid, block, smem >> > (d_a, d_h, count, r, bit_group);
		CUDA_CALL(cudaEventRecord(end));
		CUDA_CALL(cudaMemcpy(h_h2, d_h, h_total_size, cudaMemcpyDeviceToHost));
		gpu_ms = MyCudaEventElapsedTime(start, end);
		CUDA_CALL(cudaEventDestroy(end));
		CUDA_CALL(cudaEventDestroy(start));
	}

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "GPU " << gpu_ms << " ms - Speedup: x" << cpu_ms / gpu_ms << std::endl;
	#endif

	// check arrays
	CheckArrays(h_h1, h_h2, h_total_count);

	// deallocate
	CUDA_CALL(cudaFree(d_h));
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFreeHost(h_h2));
	CUDA_CALL(cudaFreeHost(h_h1));
	CUDA_CALL(cudaFreeHost(h_a));
}

void BenchmarkBuildHistogram()
{
	int counts[] =
	{
		1024 * 1024 * 64,
		1024 * 1024 * 128,
		1024 * 1024 * 256,
		1024 * 1024 * 512,
		1024 * 1024 * 1024,
	};

	int blocks[] =
	{
		64,
		128,
		256,
		512,
		1024,
	};

	int rs[] =
	{
		1,
		2,
		4,
		8,
		16,
	};

	for (int c_i = 0; c_i < MYARRAYCOUNT(counts); c_i++)
	{
		for (int b_i = 0; b_i < MYARRAYCOUNT(blocks); b_i++)
		{
			for (int r_i = 0; r_i < MYARRAYCOUNT(rs); r_i++)
			{
				RNG rng = RNG(0, 0, (32 / rs[r_i]));
				TestBuildHistogram(counts[c_i], blocks[b_i], rs[r_i], rng.Get(), 0, UINT32_MAX);
			}
		}
	}
}

#define LSD_RADIX_SORT_TEST_ELEMS_COUNT (1024 * 2)
#define LSD_RADIX_SORT_TEST_BLOCK_DIM (1024)
#define LSD_RADIX_SORT_TEST_MIN 0
#define LSD_RADIX_SORT_TEST_MAX UINT32_MAX
#define LSD_RADIX_SORT_TEST_R 4

void TestLSDRadixSort()
{
	#if 0
	#ifdef PRINT_TIMINGS
	std::cout << "-- Test LSD Radix Sort --" << std::endl;
	#endif

	RNG rng = RNG(0, LSD_RADIX_SORT_TEST_MIN, LSD_RADIX_SORT_TEST_MAX);

	// allocate
	int block_dim = LSD_RADIX_SORT_TEST_BLOCK_DIM;
	int r = LSD_RADIX_SORT_TEST_R;
	int count = LSD_RADIX_SORT_TEST_ELEMS_COUNT;
	dim3 block(block_dim);
	dim3 grid((count + block_dim - 1) / block_dim);
	int histogram_count = (1 << r);
	size_t histogram_size = histogram_count * grid.x * sizeof(uint32_t);
	size_t size = count * sizeof(uint32_t);
	uint32_t* h_a = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_b = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* h_c = (uint32_t*)MyCudaHostAlloc(size);
	uint32_t* d_a = (uint32_t*)MyCudaMalloc(size);
	uint32_t* d_b = (uint32_t*)MyCudaMalloc(size);
	uint32_t* d_histogram = (uint32_t*)MyCudaMalloc(histogram_size);
	uint32_t* histogram = (uint32_t*)calloc(1 << r, sizeof(*histogram));

	#ifdef PRINT_TIMINGS
	std::cout << "Sorting " << (double)(size) / (1024.0 * 1024.0 * 1024.0) << " GB of data" << std::endl;
	#endif

	// populate input
	for (int i = 0; i < count; i++) h_a[i] = rng.Get();

	float cpu_ms = 0.0f;
	{
		int64_t start = GetTimestamp();
		LSDRadixSort(h_a, h_b, count, histogram, r);
		int64_t end = GetTimestamp();
		cpu_ms = GetElapsedMS(start, end);
	}

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "CPU LSD Radix Sort (" << r << " bits keys): " << cpu_ms << " ms" << std::endl;
	#endif

	float gpu_ms = 0.0f;
	{
		cudaEvent_t start = MyCudaEventCreate();
		cudaEvent_t end = MyCudaEventCreate();
		CUDA_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaEventRecord(start));
		{
			// 1) Count
			// TODO: initialize smem to zero. 
			// Here smem is used as an histogram. The number of cells in the histogram
			// If histogram is smaller than the block, then some threads won't do any zero initialization
			// If histogram is as big as the block, then each thread will initialize one histogram cell
			// If histogram is bigger than the block, then each thread will initialize one or more histogram cells
			// TODO: build histogram using atomic operations on smem

			// 2) Exclusive Prefix Sum
			// TODO: Compute prefix sum on each block local histogram. This computes the block local offsets
			// TODO: Compute prefix sum on the global histogram, stored in column major order. This computes the global offsets
			// For starters, we write to global memory each block local histogram
			// Then we copy the histogram to another chunk of global memory
			// We perform a block local prefix sum on each histogram in the first chunk of global memory
			// We transpose the second chunk of global memory
			// We perform a prefix sum on the entire second chunk of global memory
			// We transpose again the second chunk of global memory
			// The two prefix sums, can be done on different streams, to help with parallelism

			// 3) Sort
			// TODO: Sort the elements by block, using smem and LSDBinaryRadixSort
			// TODO: Build a destination table using local and global offsets (random access, use SMEM for there tables)
			// Here we require a four times the SMEM. Elements + local offsets + global offsets + destination table
			// TODO: Reorder elements using destination table

			size_t smem = (1 << r) * sizeof(uint32_t);
			BuildHistogramKernel << <grid, block, smem >> > (d_a, count, d_histogram, r, 0);
		}
		CUDA_CALL(cudaEventRecord(end));
		CUDA_CALL(cudaMemcpy(h_c, d_b, size, cudaMemcpyDeviceToHost));
		gpu_ms = MyCudaEventElapsedTime(start, end);
		CUDA_CALL(cudaEventDestroy(end));
		CUDA_CALL(cudaEventDestroy(start));
	}

	// print timings
	#ifdef PRINT_TIMINGS
	std::cout << "GPU LSD Radix Sort (" << r << " bits keys): " << gpu_ms << " ms - Speedup: x" << cpu_ms / gpu_ms << std::endl;
	#endif

	// TODO: check arrays
	//CheckArrays(h_b, h_c, count);

	// deallocate
	free(histogram);
	CUDA_CALL(cudaFree(d_b));
	CUDA_CALL(cudaFree(d_a));
	CUDA_CALL(cudaFreeHost(h_c));
	CUDA_CALL(cudaFreeHost(h_b));
	CUDA_CALL(cudaFreeHost(h_a));
	#endif
}

void BenchmarkGPUPrefixSum()
{
	int count[] =
	{
		1024 * 1024 * 64,
		1024 * 1024 * 128,
		1024 * 1024 * 256,
		1024 * 1024 * 512,
		1024 * 1024 * 1024,
	};

	int threads_per_block[] =
	{
		64,
		128,
		256,
		512,
		1024,
	};

	for (int i = 0; i < MYARRAYCOUNT(threads_per_block); i++)
	{
		for (int j = 0; j < MYARRAYCOUNT(count); j++)
		{
			TestGPUPrefixSum(count[i], threads_per_block[j], 0, 10);
		}
	}
}

#define PREFIX_SUM_TEST_ELEMS_COUNT (1024 * 1024)
#define PREFIX_SUM_TEST_ELEMS_THREADS_PER_BLOCK (128)
#define PREFIX_SUM_TEST_ELEMS_MIN 0
#define PREFIX_SUM_TEST_ELEMS_MAX 10

#define BUILD_HISTOGRAM_TEST_ELEMS_COUNT (1024 * 1024)
#define BUILD_HISTOGRAM_TEST_BLOCK_DIM (256)
#define BUILD_HISTOGRAM_TEST_MIN 0
#define BUILD_HISTOGRAM_TEST_MAX 10
#define BUILD_HISTOGRAM_TEST_BIT_GROUP 0
#define BUILD_HISTOGRAM_TEST_R 4

int main()
{
	CheckForHostLeaks();

	#ifdef BENCHMARK_BUILD_HISTOGRAMS
	BenchmarkBuildHistogram();
	#else
	TestSequentialLSDRadixSort();
	TestBlockPrefixSumKernel();
	TestGPUPrefixSum(PREFIX_SUM_TEST_ELEMS_COUNT, PREFIX_SUM_TEST_ELEMS_THREADS_PER_BLOCK, PREFIX_SUM_TEST_ELEMS_MIN, PREFIX_SUM_TEST_ELEMS_MAX);
	TestLSDBinaryRadixSort();
	TestTranspose();
	TestBuildHistogram(BUILD_HISTOGRAM_TEST_ELEMS_COUNT, BUILD_HISTOGRAM_TEST_BLOCK_DIM, BUILD_HISTOGRAM_TEST_R, BUILD_HISTOGRAM_TEST_BIT_GROUP, BUILD_HISTOGRAM_TEST_MIN, BUILD_HISTOGRAM_TEST_MAX);
	TestLSDRadixSort();
	#endif

	return 0;
}
