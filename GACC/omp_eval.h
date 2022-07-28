#define DATA_TYPE double

extern "C" { void omp_evaluate(DATA_TYPE* input_pos, DATA_TYPE* input_vel, DATA_TYPE* input_mass, int n_particles, int steps, DATA_TYPE G, DATA_TYPE eps, DATA_TYPE dt, double *totalTime); }