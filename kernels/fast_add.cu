#include "kernels.h"

__global__
void fast_add_3to3(float3 *s, float3 *d, float mul){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i].x = d[i].x + s[i].x * mul;
    d[i].y = d[i].y + s[i].y * mul;
    d[i].z = d[i].z + s[i].z * mul;

}

__global__
void fast_add_4to3(float4 *s, float3 *d, float mul){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i].x = d[i].x + s[i].x * mul;
    d[i].y = d[i].y + s[i].y * mul;
    d[i].z = d[i].z + s[i].z * mul;

}