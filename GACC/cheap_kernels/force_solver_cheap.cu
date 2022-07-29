#include "kernels.h"
#include <cuda_fp16.h>


__global__
void force_solve_cheap(half2* pos, half* mass, float* acc_phi, float G, float eps, int n_particles){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;

    __half2 h2G = __float2half2_rn(G);
    __half heps = __float2half(eps);

    __half2 h2pos_ixy = pos[i*2];
    __half hpos_iz = __high2half(pos[i*2 + 1]);

    __half hmass_i = mass[i];
    __half2 h2mass_j;

    __half2 h2temp0;
    __half2 h2temp1;
    __half2 h2temp2;

    __half htemp0 = 1;
    __half htemp1 = 2;
    __half htemp2 = 3;
    __half htemp3 = 4;

    __half2 h2diffxy;
    __half hdiffz;

    __half2 h2axy = __float2half2_rn(0);
    __half2 h2az_gpe = __float2half2_rn(0);

    __half2 h2dist;
    __half2 h2Gmul;
    __half2 h2acc_mul;

    
    for (j = 0; j < n_particles; j++){

        if (j != i) {

            if (__hne(hmass_i,0)){
            
                //LOAD J_POS
                h2temp0 = pos[j*2]; //loads x,y into half2
                htemp0 = __high2half(pos[j*2 + 1]); //loads z into half

                h2mass_j = __half2half2(mass[j]); //loads mass of j into half2

                h2diffxy = __hsub2(h2temp0,h2pos_ixy); //gets difference of jxy and ixy into half2
                hdiffz = __hsub(htemp0,hpos_iz); //gets difference of jz and iz into half

                h2temp0 = __hmul2(h2diffxy,h2diffxy); //squares diffxy
                htemp0 = __hfma(hdiffz,hdiffz,heps); //squars diffz and adds smoothing eps

                htemp1 = __low2half(h2temp0); //divides xy into two halfs
                htemp2 = __high2half(h2temp0);

                htemp3 = __hadd(htemp0,htemp1); //adds (diffz**2 + eps) and diffx**2
                htemp0 = __hadd(htemp3,htemp2); //adds (diffz**2 + eps + diffx**2) and diffy**2

                h2temp1 = __half2half2(htemp0); //copies half (diffz**2 + eps + diffx**2 + diffy**2) to a half2 (distance**2)

                h2dist = h2sqrt(h2temp1); //square roots half (diffz**2 + eps + diffx**2 + diffy**2) to a half2

                h2temp0 = __hmul2(h2temp1,h2dist); //multiplies half2 distance by half2 distance**2 to get half2 distance**3

                h2Gmul = __hmul2(h2G,h2mass_j); //multiplies G by mass_j to half2
                h2acc_mul = __h2div(h2Gmul,h2temp0); //divides by distance**3

                htemp1 = __hdiv(__hmul(hmass_i,__high2half(h2Gmul)),__high2half(h2dist));

                h2temp0 = __hmul2(h2acc_mul,h2diffxy); //calculates acceleration for xy
                htemp0 = __hmul(__high2half(h2acc_mul),hdiffz); //calculates acceleration for z and multiplies (G * mj / d**3) by mass_i * d**2 for gpe

                h2axy = __hadd2(h2axy,h2temp0); //adds acceleration of xy to h2axy
                h2az_gpe = __hadd2(h2az_gpe,__halves2half2(htemp0,htemp1)); //adds acceleation of z and gpe to h2az_gpe
            }

        }

    }
    

    acc_phi[i*4] = __low2float(h2axy);
    acc_phi[i*4 + 1] = __high2float(h2axy);
    acc_phi[i*4 + 2] = __low2float(h2az_gpe);
    acc_phi[i*4 + 3] = (-1) * __high2float(h2az_gpe);
    
}