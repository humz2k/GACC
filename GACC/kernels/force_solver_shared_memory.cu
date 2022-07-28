#include "kernels.h"

__global__
void force_solve_shared_mem(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles){

     //TODO: use local memory: make shared memory array of size blocksize. 

    
    extern __shared__ float s[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    int k;

    float pos_ix = pos[i*3];
    float pos_iy = pos[i*3 + 1];
    float pos_iz = pos[i*3 + 2];
    float mass_i = mass[i];

    float pos_kx;
    float pos_ky;
    float pos_kz;
    float mass_k;

    float diffx;
    float diffy;
    float diffz;

    float ax = 0;
    float ay = 0;
    float az = 0;
    float gpe = 0;

    float dist;

    float acc_mul;

    int n_repeats = n_particles / blockDim.x;
    int laneID = threadIdx.x % 32;

    for (j = 0; j < n_repeats; j++){

        int startIdx = j * blockDim.x;
        int myIdx = threadIdx.x + startIdx;
        s[threadIdx.x * 4] = pos[myIdx * 3];
        s[threadIdx.x * 4 + 1] = pos[myIdx * 3 + 1];
        s[threadIdx.x * 4 + 2] = pos[myIdx * 3 + 2];
        s[threadIdx.x * 4 + 3] = mass[myIdx];

        __syncthreads();

        for (k = 0; k < blockDim.x; k++){

            if (k + startIdx != i){

                if (mass_i != 0){

                    pos_kx = s[k*4];
                    pos_ky = s[k*4 + 1];
                    pos_kz = s[k*4 + 2];
                    mass_k = s[k*4 + 3];

                    diffx = pos_kx - pos_ix;
                    diffy = pos_ky - pos_iy;
                    diffz = pos_kz - pos_iz;

                    dist = sqrt((diffx*diffx) + (diffy*diffy) + (diffz*diffz) + eps);

                    acc_mul = G * mass_k / (dist * dist * dist);
                    ax = ax + diffx * acc_mul;
                    ay = ay + diffy * acc_mul;
                    az = az + diffz * acc_mul;

                    gpe = gpe + (-1) * G * mass_k * mass_i / dist;
                }

            }

        }

    }

    acc_phi[i*4] = ax;
    acc_phi[i*4 + 1] = ay;
    acc_phi[i*4 + 2] = az;
    acc_phi[i*4 + 3] = gpe;
    
}