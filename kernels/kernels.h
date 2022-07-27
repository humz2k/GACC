extern __global__ void force_solve_gpu(float3* pos, float* mass, float4* acc_phi, float G, float eps, int n_particles);

extern __global__ void fast_add_3to3(float3 *s, float3 *d, float mul);

extern __global__ void fast_add_4to3(float4 *s, float3 *d, float mul);