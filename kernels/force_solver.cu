#include "kernels.h"

__global__
void force_solve_gpu(float3* pos, float* mass, float4* acc_phi, float G, float eps, int n_particles){

     //TODO: use local memory: make shared memory array of size blocksize. 

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;

    float3 pos_i;
    pos_i.x = pos[i].x;
    pos_i.y = pos[i].y;
    pos_i.z = pos[i].z;
    float mass_i = mass[i];

    float3 diff;

    float3 pos_j;
    float mass_j;

    float4 out_acc_phi;

    float temp0;
    float temp1;
    float temp2;

    float dist;

    for (j = 0; j < n_particles; j++){

        if (i != j){

            pos_j.x = pos[j].x;
            pos_j.y = pos[j].y;
            pos_j.z = pos[j].z;
            mass_j = mass[j];

            diff.x = pos_j.x - pos_i.x;
            diff.y = pos_j.y - pos_i.y;
            diff.z = pos_j.z - pos_i.z;

            temp1 = eps;

            temp0 = diff.x * diff.x;
            temp1 += temp0;
            temp0 = diff.y * diff.y;
            temp1 += temp0;
            temp0 = diff.z * diff.z;
            temp1 += temp0;

            dist = sqrt(temp1);

            if (dist != 0){

                temp0 = dist * dist * dist;
                temp2 = G * mass_j;
                temp1 = temp2 / temp0;

                temp0 = diff.x * temp1;
                out_acc_phi.x += temp0;

                temp0 = diff.y * temp1;
                out_acc_phi.y += temp0;

                temp0 = diff.z * temp1;
                out_acc_phi.z += temp0;

                temp0 = -G;
                temp1 = mass_j * mass_i;
                temp2 = temp0 * temp1;

                out_acc_phi.w += temp2 / dist;
            }

        }
    }

    //acc_phi[i].x = out_acc_phi.x;
    //acc_phi[i].y = out_acc_phi.y;
    //acc_phi[i].z = out_acc_phi.z;
    //acc_phi[i].w = out_acc_phi.w;

}