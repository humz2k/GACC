//eval.h

extern "C" { void c_evaluate(float* input_pos, float* input_vel, int n_particles, int steps, float G, float eps, float dt, int n_params); }

extern "C" { void printArray(float* array, int length); }