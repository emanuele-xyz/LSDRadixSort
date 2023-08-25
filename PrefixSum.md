# Prefix Sum

* TODO: make it a nice table
* We can observe that 128 is a good number of blocks

-- Test exclusive prefix sum --
Prefix sum of 0.25 GB of data
Threads per block: 64
Prefix Sum Sequential: 146.427 ms
GPU Prefix Sum: 33.8891 ms
Speedup: x4.32078

-- Test exclusive prefix sum --
Prefix sum of 0.25 GB of data
Threads per block: 128
Prefix Sum Sequential: 146.482 ms
GPU Prefix Sum: 23.8254 ms
Speedup: x6.14815

-- Test exclusive prefix sum --
Prefix sum of 0.25 GB of data
Threads per block: 256
Prefix Sum Sequential: 146.433 ms
GPU Prefix Sum: 26.9998 ms
Speedup: x5.42349

-- Test exclusive prefix sum --
Prefix sum of 0.25 GB of data
Threads per block: 512
Prefix Sum Sequential: 146.913 ms
GPU Prefix Sum: 122.027 ms
Speedup: x1.20393

-- Test exclusive prefix sum --
Prefix sum of 0.25 GB of data
Threads per block: 1024
Prefix Sum Sequential: 146.977 ms
GPU Prefix Sum: 194.226 ms
Speedup: x0.75673

-- Test exclusive prefix sum --
Prefix sum of 0.5 GB of data
Threads per block: 64
Prefix Sum Sequential: 293.517 ms
GPU Prefix Sum: 192.826 ms
Speedup: x1.52218

-- Test exclusive prefix sum --
Prefix sum of 0.5 GB of data
Threads per block: 128
Prefix Sum Sequential: 293.081 ms
GPU Prefix Sum: 179.034 ms
Speedup: x1.63701

-- Test exclusive prefix sum --
Prefix sum of 0.5 GB of data
Threads per block: 256
Prefix Sum Sequential: 293.583 ms
GPU Prefix Sum: 185.757 ms
Speedup: x1.58047

-- Test exclusive prefix sum --
Prefix sum of 0.5 GB of data
Threads per block: 512
Prefix Sum Sequential: 293.741 ms
GPU Prefix Sum: 202.142 ms
Speedup: x1.45314

-- Test exclusive prefix sum --
Prefix sum of 0.5 GB of data
Threads per block: 1024
Prefix Sum Sequential: 293.649 ms
GPU Prefix Sum: 222.906 ms
Speedup: x1.31737

-- Test exclusive prefix sum --
Prefix sum of 1 GB of data
Threads per block: 64
Prefix Sum Sequential: 593.379 ms
GPU Prefix Sum: 149.154 ms
Speedup: x3.9783

-- Test exclusive prefix sum --
Prefix sum of 1 GB of data
Threads per block: 128
Prefix Sum Sequential: 593.506 ms
GPU Prefix Sum: 98.7862 ms
Speedup: x6.00798

-- Test exclusive prefix sum --
Prefix sum of 1 GB of data
Threads per block: 256
Prefix Sum Sequential: 593.162 ms
GPU Prefix Sum: 108.772 ms
Speedup: x5.45328

-- Test exclusive prefix sum --
Prefix sum of 1 GB of data
Threads per block: 512
Prefix Sum Sequential: 591.757 ms
GPU Prefix Sum: 122.808 ms
Speedup: x4.81854

-- Test exclusive prefix sum --
Prefix sum of 1 GB of data
Threads per block: 1024
Prefix Sum Sequential: 592.13 ms
GPU Prefix Sum: 196.435 ms
Speedup: x3.01438

-- Test exclusive prefix sum --
Prefix sum of 2 GB of data
Threads per block: 64
Prefix Sum Sequential: 1194.59 ms
GPU Prefix Sum: 247.758 ms
Speedup: x4.82162

-- Test exclusive prefix sum --
Prefix sum of 2 GB of data
Threads per block: 128
Prefix Sum Sequential: 1190.93 ms
GPU Prefix Sum: 188.951 ms
Speedup: x6.30286

-- Test exclusive prefix sum --
Prefix sum of 2 GB of data
Threads per block: 256
Prefix Sum Sequential: 1297.38 ms
GPU Prefix Sum: 247.583 ms
Speedup: x5.24016

-- Test exclusive prefix sum --
Prefix sum of 2 GB of data
Threads per block: 512
Prefix Sum Sequential: 1186.87 ms
GPU Prefix Sum: 241.033 ms
Speedup: x4.92408

-- Test exclusive prefix sum --
Prefix sum of 2 GB of data
Threads per block: 1024
Prefix Sum Sequential: 1197.76 ms
GPU Prefix Sum: 365.554 ms
Speedup: x3.27658

-- Test exclusive prefix sum --
Prefix sum of 4 GB of data
Threads per block: 64
Prefix Sum Sequential: 2432.9 ms
GPU Prefix Sum: 474.274 ms
Speedup: x5.12974

-- Test exclusive prefix sum --
Prefix sum of 4 GB of data
Threads per block: 128
Prefix Sum Sequential: 2376.2 ms
GPU Prefix Sum: 353.804 ms
Speedup: x6.71615

-- Test exclusive prefix sum --
Prefix sum of 4 GB of data
Threads per block: 256
Prefix Sum Sequential: 2353.85 ms
GPU Prefix Sum: 431.149 ms
Speedup: x5.45947

-- Test exclusive prefix sum --
Prefix sum of 4 GB of data
Threads per block: 512
Prefix Sum Sequential: 2356.89 ms
GPU Prefix Sum: 437.856 ms
Speedup: x5.38278

-- Test exclusive prefix sum --
Prefix sum of 4 GB of data
Threads per block: 1024
Prefix Sum Sequential: 2368.12 ms
GPU Prefix Sum: 755.682 ms
Speedup: x3.13375
