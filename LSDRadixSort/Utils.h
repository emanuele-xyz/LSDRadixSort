#pragma once

#include <random>
#include <stdio.h>

#ifdef NDEBUG
#define MYCRASH() (*((int*)(0)) = 0)
#else
#define MYCRASH() __debugbreak()
#endif
#define MYASSERT(p) do { if (!(p)) MYCRASH(); } while (0)

#define MYARRAYCOUNT(a) (sizeof(a) / sizeof(*(a)))

class RNG
{
public:
	RNG(unsigned seed, uint32_t min, uint32_t max);
public:
	uint32_t Get() { return m_distribution(m_engine); }
private:
	std::default_random_engine m_engine;
	std::uniform_int_distribution<uint32_t> m_distribution;
};

void CheckForHostLeaks();
int64_t GetTimerFrequency();
int64_t GetTimestamp();
float GetElapsedMS(int64_t start, int64_t end);
void CheckArrays(uint32_t* a, uint32_t* b, int count);
void PrintArray(char label, uint32_t* a, int count);
void PrintMatrix(uint32_t* a, int m, int n);