#include "kernels.h"
#include <cuda_fp16.h>

__global__
void force_solve_cheap_sqrt(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = 0;

    float pos_ix = pos[i*3];
    float pos_iy = pos[i*3 + 1];
    float pos_iz = pos[i*3 + 2];

    float pos_jx;
    float pos_jy;
    float pos_jz;

    float diffx;
    float diffy;
    float diffz;

    __half hdiffx;
    __half hdiffy;
    __half hdiffz;

    float ax = 0;
    float ay = 0;
    float az = 0;
    float gpe = 0;

    float mass_i = mass[i];
    float mass_j;

    float dist;

    float acc_mul;

    __half htemp0;
    __half htemp1;
    __half htemp2;
    __half htemp3;

    float temp0;
    float temp1;
    
    for (j = 0; j < n_particles; j++){

        if (j != i) {
            pos_jx = pos[j*3];
            pos_jy = pos[j*3 + 1];
            pos_jz = pos[j*3 + 2];

            mass_j = mass[j];

            diffx = pos_jx - pos_ix;
            diffy = pos_jy - pos_iy;
            diffz = pos_jz - pos_iz;

            hdiffx = __float2half(diffx);
            hdiffy = __float2half(diffy);
            hdiffz = __float2half(diffz);

            htemp0 = __hmul(hdiffx,hdiffx);
            htemp1 = __hmul(hdiffy,hdiffy);
            htemp2 = __hmul(hdiffz,hdiffz);

            htemp3 = __hadd(htemp0,htemp1);
            htemp0 = __hadd(htemp3,htemp2);

            htemp2 = hsqrt(htemp0);
            dist = __half2float(htemp2);

            temp0 = dist*dist;
            temp1 = temp0 * dist;
            temp0 = G * mass_j;

            acc_mul = temp0/(temp1);

            temp0 = acc_mul * diffx;
            ax = ax + temp0;

            temp0 = acc_mul * diffy;
            ay = ay + temp0;

            temp0 = acc_mul * diffz;
            az = az + temp0;

            temp0 = mass_i * mass_j;
            temp1 = temp0 * G;
            temp0 = temp1 * -1;

            gpe = gpe + (temp0 / dist);
        }

    }

    acc_phi[i*4] = ax;
    acc_phi[i*4 + 1] = ay;
    acc_phi[i*4 + 2] = az;
    acc_phi[i*4 + 3] = gpe;
    
}