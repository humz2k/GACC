#include "kernels.h"

__global__
void force_solve_gpu(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles){

     //TODO: use local memory: make shared memory array of size blocksize. 

    extern __shared__ float s[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = 0;

    float pos_ix;
    float pos_iy;
    float pos_iz;

    float diffx;
    float diffy;
    float diffz;

    float pos_jx;
    float pos_jy;
    float pos_jz;

    float ax = 0;
    float ay = 0;
    float az = 0;
    float gpe = 0;

    float mass_i = mass[i];
    float mass_j = 0;

    float dist = 0;
    float acc_mul = 0;

    for (j = 0; j < n_particles; j++){

        if (j != i) {
            pos_jx = pos[j*3];
            pos_jy = pos[j*3 + 1];
            pos_jz = pos[j*3 + 2];

            mass_j = mass[j];
            
            diffx = pos_jx - pos_ix;
            diffy = pos_jy - pos_iy;
            diffz = pos_jz - pos_iz;

            dist = sqrt((diffx*diffx) + (diffy*diffy) + (diffz*diffz) + eps);

            acc_mul = G * mass_j / (dist * dist * dist);
            ax += diffx * acc_mul;
            ay += diffy * acc_mul;
            az += diffz * acc_mul;

            gpe += (-1) * G * mass_j * mass_i / dist;
        }

    }

    acc_phi[i*4] = ax;
    acc_phi[i*4 + 1] = ay;
    acc_phi[i*4 + 2] = az;
    acc_phi[i*4 + 3] = gpe;
}