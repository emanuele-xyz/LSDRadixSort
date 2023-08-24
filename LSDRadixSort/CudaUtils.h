#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"

#define PRINT_CUDA_ERROR(err) printf("%s(%d) [CUDA]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err))
#define CUDA_CALL(call) do { cudaError_t err = (call); if (err != cudaSuccess) { PRINT_CUDA_ERROR(err); MYASSERT(false); } } while (0)

void* MyCudaHostAlloc(size_t size, unsigned flags = 0);
void* MyCudaMalloc(size_t size);
cudaEvent_t MyCudaEventCreate();
float MyCudaEventElapsedTime(cudaEvent_t start, cudaEvent_t end);
