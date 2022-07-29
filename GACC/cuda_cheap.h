//eval.h

#include <cuda_fp16.h>

extern "C" { void cuda_evaluate(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, int n_params, int solver, int v, double *saveTime, double *totalTime, double *copyTime); }