#include <cuda_fp16.h>

extern __global__ void fast_add_3to3(float *s, float *d, float mul);

extern __global__ void fast_add_4to3(float *s, float *d, float mul);

extern __global__ void force_solve_cheap(half2* pos, half* mass, float* acc_phi, float G, float eps, int n_particles);

extern __global__ void force_solve_cheap_shared_mem(half2* pos, half* mass, float* acc_phi, float G, float eps, int n_particles);

extern __global__ void copyFloat2Half(float *s, half *d);

extern __global__ void copyHalf2Float(half *s, float *d);

extern __global__ void copyFloat2Half_dim1(float *s, half *d);

extern __global__ void copyFloat2Half2(float *s, half2 *d);