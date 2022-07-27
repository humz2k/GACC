extern __global__ void force_solve_gpu(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles);

extern __global__ void fast_add_3to3(float *s, float *d, float mul);

extern __global__ void fast_add_4to3(float *s, float *d, float mul);