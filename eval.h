//eval.h

extern "C" { void c_evaluate(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, int n_params); }

extern "C" { void printArray(float* array, int length); }