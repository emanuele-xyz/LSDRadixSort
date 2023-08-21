#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <bitset>

// Windows only
#include <intrin.h>

class RNG
{
public:
	RNG(unsigned seed, uint32_t min, uint32_t max)
		: m_engine{ seed }
		, m_distribution(min, max)
	{}
public:
	uint32_t Get() { return m_distribution(m_engine); }
private:
	std::default_random_engine m_engine;
	std::uniform_int_distribution<uint32_t> m_distribution;
};

/*
* Extract the i-th group of r bits from n
*/
#define GET_R_BITS(n, r, i) (((1 << r) - 1) & (n >> (i * r)))

/*
* in:    input array (will be modified)
* out:   output array
* count: number of elements to sort
* r:     number of bits to consider as keys
*/
void LSDRadixSort(uint32_t* in, uint32_t* out, int count, int* histogram, int r)
{
	int iterations = (sizeof(*in) * 8) / r;
	for (int i = 0; i < iterations; i++)
	{
		// iter 1 -- first 4 bits
		// 141       152       183       216       154       220       140       217       108       160
		// 1101      1000      0111      1000      1010      1100      1100      1001      1100      0000
		// 
		// 160       183       152       216       217       154       220       140       108       141
		// 0000      0111      1000      1000      1001      1010      1100      1100      1100      1101

		// iter 2 -- second 4 bits
		// 160       183       152       216       217       154       220       140       108       141
		// 1010      1011      1001      1101      1101      1001      1101      1000      0110      1000
		// 
		// 108       140       141       152       154       160       183       216       217       220
		// 0110      1000      1000      1001      1001      1010      1011      1101      1101      1101

		memset(histogram, 0, sizeof(*histogram) * (1 << r));

		// build histogram
		for (int j = 0; j < count; j++)
		{
			uint32_t val = in[j];
			int key = GET_R_BITS(val, r, i);
			histogram[key]++;
		}

		// 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 3, 1, 0, 0
		// 1, 1, 1, 1, 1, 1, 1, 2, 4, 5, 6, 6, 9, 10, 10, 10
		
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

#define ELEMS_COUNT 10
#define ELEMS_MIN 0
#define ELEMS_MAX 1024
#define ELEMS_R 4

#ifdef NDEBUG
#define MYCRASH() (*((int*)(0)) = 0)
#else
#define MYCRASH() __debugbreak()
#endif
#define MYASSERT(p) do { if (!(p)) MYCRASH(); } while(false)

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

int main()
{
	RNG rng = RNG(0, ELEMS_MIN, ELEMS_MAX);

	int count = ELEMS_COUNT;
	uint32_t* a = (uint32_t*)(calloc(count, sizeof(*a)));
	uint32_t* b = (uint32_t*)(calloc(count, sizeof(*b)));
	uint32_t* c = (uint32_t*)(calloc(count, sizeof(*c)));
	int* h = (int*)(calloc(1 << ELEMS_R, sizeof(*h)));

	// populate a and c
	for (int i = 0; i < count; i++)
	{
		int elem = rng.Get();
		a[i] = elem;
		c[i] = elem;
	}

	// sort a writing result in b using LSD radix sort
	LSDRadixSort(a, b, count, h, ELEMS_R);

	// sort c using standard library sort
	std::sort(c, c + count);

	PrintArray('b', b, count);
	PrintArray('c', c, count);

	CheckArrays(b, c, count);

	free(h);
	free(c);
	free(b);
	free(a);

	return 0;
}
