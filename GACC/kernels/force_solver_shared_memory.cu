#include "kernels.h"

__global__
void force_solve_shared_mem(DATA_TYPE* pos, DATA_TYPE* mass, DATA_TYPE* acc_phi, DATA_TYPE G, DATA_TYPE eps, int n_particles){

     //TODO: use local memory: make shared memory array of size blocksize. 

    
    extern __shared__ DATA_TYPE s[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    int k;

    DATA_TYPE pos_ix = pos[i*3];
    DATA_TYPE pos_iy = pos[i*3 + 1];
    DATA_TYPE pos_iz = pos[i*3 + 2];
    DATA_TYPE mass_i = mass[i];

    DATA_TYPE pos_kx;
    DATA_TYPE pos_ky;
    DATA_TYPE pos_kz;
    DATA_TYPE mass_k;

    DATA_TYPE diffx;
    DATA_TYPE diffy;
    DATA_TYPE diffz;

    DATA_TYPE ax = 0;
    DATA_TYPE ay = 0;
    DATA_TYPE az = 0;
    DATA_TYPE gpe = 0;

    DATA_TYPE dist;

    DATA_TYPE acc_mul;

    int n_repeats = n_particles / blockDim.x;
    //int laneID = threadIdx.x % 32;

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