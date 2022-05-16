# DGEMM on KNL

## Overview

Include DGEMM code tunned for KNL. The purpose of this code is to demonstrate how to tune a general GEMM algorithm for a specific hardware (i.e. Intel KNL). Thus the implemented GEMM do not cover the case where input is not multiply of kernel size.


The implemented DGEMM is row major. The implementation mostly based on [1] and [2] with further modification based on the hyper-parameter choice. A more detailed explaination of the implementation can be found [here](./report.pdf)



## Notice to UC Berkeley CS267 student
You should **NOT** using this code (in part or whole) for your GEMM assignment. A modifed copy of this repo has been submitted as student homework to ensure the autograder have a copy of this repo. Using part or whole of this code is considered as violation of UC Berkeley Student Honor Code and would be reported to Center for Student Conduct.



## Build the code
```shell
module load cmake
# Use intel compiler
module swap PrgEnv-gnu PrgEnv-intel
# Use GNU compiler
module swap PrgEnv-intel PrgEnv-gnu
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
