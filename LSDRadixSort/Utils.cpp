#include "Utils.h"

#include <iostream>

#if defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <time.h>
#endif

RNG::RNG(unsigned seed, uint32_t min, uint32_t max)
	: m_engine{ seed }
	, m_distribution(min, max)
{}

void CheckForHostLeaks()
{
  #if defined(_MSC_VER)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
  #endif
}

#if defined(_MSC_VER)
int64_t GetTimerFrequency()
{
	LARGE_INTEGER freq = {};
	QueryPerformanceFrequency(&freq);
	return freq.QuadPart;
}

int64_t GetTimestamp()
{
	LARGE_INTEGER time = {};
	QueryPerformanceCounter(&time);
	return time.QuadPart;
}

float GetElapsedMS(int64_t start, int64_t end)
{
	static int64_t frequency = GetTimerFrequency();
	float dt = (float)(end - start) / (float)(frequency);
	return dt * 1000.0f;
}
#else
int64_t GetTimerFrequency()
{
  return (int64_t)(time(NULL));
}

int64_t GetTimestamp()
{
  return (int64_t)(time(NULL));
}

float GetElapsedMS(int64_t start, int64_t end)
{
  return (float)(end - start) * 1000.0f;
}
#endif

void CheckArrays(uint32_t* a, uint32_t* b, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		MYASSERT(a[i] == b[i]);
	}
}

void CheckIfSorted(uint32_t* a, int count, int r, int bit_group)
{
	for (int i = 1; i < count; i++)
	{
		uint32_t prev = a[i - 1];
		uint32_t curr = a[i];
		uint32_t prev_val = GET_R_BITS(prev, r, bit_group);
		uint32_t curr_val = GET_R_BITS(curr, r, bit_group);
		MYASSERT(prev <= curr);
	}
}

void PrintArray(char label, uint32_t* a, int count)
{
	std::cout << label << ": ";
	for (int i = 0; i < count; i++)
	{
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}

void PrintMatrix(uint32_t* a, int m, int n)
{
	for (int row = 0; row < m; row++)
	{
		for (int col = 0; col < n; col++)
		{
			std::cout << a[row * n + col] << " ";
		}
		std::cout << std::endl;
	}
}
