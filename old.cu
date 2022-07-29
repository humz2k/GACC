#include "kernels.h"
#include <cuda_fp16.h>

__global__
void force_solve_gpu(float* pos, float* mass, float* acc_phi, float G, float eps, int n_particles){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = 0;

    __half epshalf = __float2half(eps);
    __half Ghalf = __float2half(G);
    
    __half pos_ix = __float2half(pos[i*3]);
    __half pos_iy = __float2half(pos[i*3 + 1]);
    __half pos_iz = __float2half(pos[i*3 + 2]);

    __half diffx;
    __half diffy;
    __half diffz;

    __half pos_jx;
    __half pos_jy;
    __half pos_jz;

    __half ax = __float2half(0);
    __half ay = __float2half(0);
    __half az = __float2half(0);
    __half gpe = __float2half(0);

    __half mass_i = __float2half(mass[i]);
    __half mass_j = __float2half(0);

    __half dist = __float2half(0);
    __half acc_mul = __float2half(0);

    __half temp0;
    __half temp1;
    __half temp2;
    __half temp3;

    
    for (j = 0; j < n_particles; j++){

        if (j != i) {
            pos_jx = __float2half(pos[j*3]);
            pos_jy = __float2half(pos[j*3 + 1]);
            pos_jz = __float2half(pos[j*3 + 2]);

            mass_j = __float2half(mass[j]);

            diffx = __hsub (pos_jx, pos_ix);
            diffy = __hsub (pos_jy, pos_iy);
            diffz = __hsub (pos_jz, pos_iz);

            temp0 = __hmul(diffx,diffx);
            temp1 = __hmul(diffy,diffy);
            temp2 = __hmul(diffz,diffz);

            temp3 = __hadd(temp0,temp1);
            temp0 = __hadd(temp3,temp2);

            dist = hsqrt(temp0);

            temp0 = __hmul(Ghalf,mass_j);
            temp1 = __hmul(dist,dist);
            temp2 = __hmul(temp1,dist);

            acc_mul = __hdiv(temp0,temp1);
            temp0 = __hmul(diffx,acc_mul);
            ax = __hadd(ax,temp0);
            temp0 = __hmul(diffy,acc_mul);
            ay = __hadd(ay,temp0);
            temp0 = __hmul(diffz,acc_mul);
            az = __hadd(az,temp0);

            gpe = __hadd(gpe,__hdiv(__hmul(__hneg(Ghalf),__hmul(mass_j,mass_i)),dist));
        }

    }

    acc_phi[i*4] = __half2float(ax);
    acc_phi[i*4 + 1] = __half2float(ay);
    acc_phi[i*4 + 2] = __half2float(az);
    acc_phi[i*4 + 3] = __half2float(gpe);
    
}