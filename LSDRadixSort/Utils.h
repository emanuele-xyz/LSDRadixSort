#pragma once

#include <random>
#include <stdio.h>

#ifdef NDEBUG
#define MYCRASH() (*((int*)(0)) = 0)
#else
#if defined(_MSC_VER)
#define MYCRASH() __debugbreak()
#else
#define MYCRASH() (*((int*)(0)) = 0)
#endif
#endif
#define MYASSERT(p) do { if (!(p)) MYCRASH(); } while (0)

#define MYARRAYCOUNT(a) (sizeof(a) / sizeof(*(a)))

/*
* Extract the i-th group of r bits from n
*/
#define GET_R_BITS(n, r, i) (((1 << r) - 1) & (n >> (i * r)))

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
void CheckArrays(uint32_t* a, uint32_t* b, size_t count);
void CheckIfSorted(uint32_t* a, int count, int r, int bit_group);
void PrintArray(char label, uint32_t* a, int count);
void PrintMatrix(uint32_t* a, int m, int n);
