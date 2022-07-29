#include "kernels.h"
#include <cuda_fp16.h>

__global__
void copyFloat2Half(float *s, half *d){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i*3] = __float2half(s[i*3]);
    d[i*3 + 1] = __float2half(s[i*3 + 1]);
    d[i*3 + 2] = __float2half(s[i*3 + 2]);

}

__global__
void copyFloat2Half2(float *s, half2 *d){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i*2] = __floats2half2_rn(s[i*3],s[i*3+1]);
    d[i*2 + 1] = __float2half2_rn(s[i*3 + 2]);

}

__global__
void copyFloat2Half_dim1(float *s, half *d){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i] = __float2half(s[i]);

}

__global__
void copyHalf2Float(half *s, float *d){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i*3] = __half2float(s[i*3]);
    d[i*3 + 1] = __half2float(s[i*3 + 1]);
    d[i*3 + 2] = __half2float(s[i*3 + 2]);

}