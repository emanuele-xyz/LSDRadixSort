#include "CudaUtils.h"

void* MyCudaHostAlloc(size_t size, unsigned flags)
{
	void* tmp = nullptr;
	CUDA_CALL(cudaHostAlloc(&tmp, size, flags));
	return tmp;
}

void* MyCudaMalloc(size_t size)
{
	void* tmp = nullptr;
	CUDA_CALL(cudaMalloc(&tmp, size));
	return tmp;
}

cudaEvent_t MyCudaEventCreate()
{
	cudaEvent_t tmp{};
	CUDA_CALL(cudaEventCreate(&tmp));
	return tmp;
}

float MyCudaEventElapsedTime(cudaEvent_t start, cudaEvent_t end)
{
	float tmp = 0;
	CUDA_CALL(cudaEventElapsedTime(&tmp, start, end));
	return tmp;
}
