#define DATA_TYPE float

extern __global__ void force_solve_gpu(DATA_TYPE* pos, DATA_TYPE* mass, DATA_TYPE* acc_phi, DATA_TYPE G, DATA_TYPE eps, int n_particles);

extern __global__ void force_solve_shared_mem(DATA_TYPE* pos, DATA_TYPE* mass, DATA_TYPE* acc_phi, DATA_TYPE G, DATA_TYPE eps, int n_particles);

extern __global__ void fast_add_3to3(DATA_TYPE *s, DATA_TYPE *d, DATA_TYPE mul);

extern __global__ void fast_add_4to3(DATA_TYPE *s, DATA_TYPE *d, DATA_TYPE mul);