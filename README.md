# DGEMM on KNL

## Overview

Include DGEMM code tunned for KNL. The purpose of this code is to demonstrate how to tune a general GEMM algorithm for a specific hardware (i.e. Intel KNL). Thus the implemented GEMM do not cover the case where input is not multiply of kernel size.


The implemented DGEMM is row major. The implementation mostly follow [1]. However, we failed to achieve near MKL performence like [1] does. Author of [1] does not mentioned how they implemente their packing. We think part of us lower performence related to how packing is done

One should also notice, the `Algoritm 1` used in [1] corresbonding to `Figure 8` in [2], which is designed for col major order. This make packing unable to use SIMD transpose (i.e. If use for col major order and n_r is multiply of 8, then pack B can be done with SIMD transpose).

| size                           | our imp % of peak | mkl imp % of peak | percent of MKL |
| ------------------------------ | ----------------- | ----------------- | -------------- |
| m (2418) x k (3264) x n (2400) | 56.18             | 73.86             | 76.06%         |
| m (2418) x k (3264) x n (2400) | 56.10             | 73.92             | 75.90%         |
| m (4836) x k (4896) x n (4840) | 56.99             | 75.12             | 75.87%         |



## Notice to UC Berkeley CS267 student
You should **NOT** using this code (in part or whole) for your GEMM assignment.Using part or whole of this code is considered as violation of UC Berkeley Student Honor Code and would be reported to Center for Student Conduct.



## Build the code
```shell
module load cmake
# Use GNU compiler
module swap PrgEnv-intel PrgEnv-gnu
# Use Intel compiler
module swap PrgEnv-gnu PrgEnv-intel 
mkdir build && cd build
make -j 10
```

## Submit job
```shell
# submit job
sbatch job-blas
sbatch job-blocked

# check queue
sqs

# Interactive session
salloc -N 1 -C knl -q interactive -t 01:00:00
```


## Reference
1. An implementation of matrix–matrix multiplication on the Intel KNL processor with AVX-512
2. Anatomy of High-Performance Matrix Multiplication
3. UC Berkeley CS267 Homework1
