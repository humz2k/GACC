#include "kernels.h"

__global__
void force_solve_gpu(DATA_TYPE* pos, DATA_TYPE* mass, DATA_TYPE* acc_phi, DATA_TYPE G, DATA_TYPE eps, int n_particles){

     //TODO: use local memory: make shared memory array of size blocksize. 

    /*
    extern __shared__ float s[];

    int n_repeats = (n_particles + 32 - 1) / 32;
    int laneID = threadIdx.x % 32;

    for (int i = 0; i < n_repeats; i++){

    }*/
    
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = 0;

    
    DATA_TYPE pos_ix = pos[i*3];
    DATA_TYPE pos_iy = pos[i*3 + 1];
    DATA_TYPE pos_iz = pos[i*3 + 2];

    DATA_TYPE diffx;
    DATA_TYPE diffy;
    DATA_TYPE diffz;

    DATA_TYPE pos_jx;
    DATA_TYPE pos_jy;
    DATA_TYPE pos_jz;

    DATA_TYPE ax = 0;
    DATA_TYPE ay = 0;
    DATA_TYPE az = 0;
    DATA_TYPE gpe = 0;

    DATA_TYPE mass_i = mass[i];
    DATA_TYPE mass_j = 0;

    DATA_TYPE dist = 0;
    DATA_TYPE acc_mul = 0;

    

    for (j = 0; j < n_particles; j++){

        if (j != i) {

            if (mass_i != 0){
                pos_jx = pos[j*3];
                pos_jy = pos[j*3 + 1];
                pos_jz = pos[j*3 + 2];

                mass_j = mass[j];
                
                diffx = pos_jx - pos_ix;
                diffy = pos_jy - pos_iy;
                diffz = pos_jz - pos_iz;

                dist = sqrt((diffx*diffx) + (diffy*diffy) + (diffz*diffz) + eps);

                acc_mul = G * mass_j / (dist * dist * dist);
                ax = ax + diffx * acc_mul;
                ay = ay + diffy * acc_mul;
                az = az + diffz * acc_mul;

                gpe = gpe + (-1) * G * mass_j * mass_i / dist;
            }
        }

    }
    
    acc_phi[i*4] = ax;
    acc_phi[i*4 + 1] = ay;
    acc_phi[i*4 + 2] = az;
    acc_phi[i*4 + 3] = gpe;
    
}