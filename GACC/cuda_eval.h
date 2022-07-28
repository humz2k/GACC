//eval.h

#define _DATA_TYPE float

extern "C" { void cuda_evaluate(_DATA_TYPE* input_pos, _DATA_TYPE* input_vel, _DATA_TYPE* input_mass, int n_particles, int steps, _DATA_TYPE G, _DATA_TYPE eps, _DATA_TYPE dt, int n_params, int solver, int v, double *saveTime, double *totalTime, double *copyTime); }