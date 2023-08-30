#include "Utils.h"

#include <iostream>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

RNG::RNG(unsigned seed, uint32_t min, uint32_t max)
	: m_engine{ seed }
	, m_distribution(min, max)
{}

void CheckForHostLeaks()
{
	// Windows only
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
}

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

void CheckArrays(uint32_t* a, uint32_t* b, int count)
{
	for (int i = 0; i < count; i++)
	{
		MYASSERT(a[i] == b[i]);
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
