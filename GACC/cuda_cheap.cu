#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "cuda_cheap.h"
#include "cheap_kernels/kernels.h"
#include <cuda_fp16.h>

using namespace std;

void save(float* pos, float* vel, float* phi_acc, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){

        out.write( reinterpret_cast<char*>( &pos[i*3]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 1]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 2]), sizeof( float ));

        out.write( reinterpret_cast<char*>( &vel[i*3]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 1]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 2]), sizeof( float ));

        out.write( reinterpret_cast<char*>( &phi_acc[i*4]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 1]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 2]), sizeof( float ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 3]), sizeof( float ));

    }

}

extern "C" { 
    void cuda_evaluate(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, int n_params, int solver, int v, double *saveTime, double *totalTime, double *copyTime){

        double first,second;
        double total_first,total_second;

        cudaFree(0);
        cudaDeviceSynchronize();

        total_first = omp_get_wtime();

        *saveTime = 0;
        *totalTime = 0;
        *copyTime = 0;

        std::ofstream out;
        out.open( "out.dat", std::ios::out | std::ios::binary);
        std::ofstream &fp = out;

        int blockSize = n_particles;
        if (blockSize > 256){
            
            if ((n_particles % 256) == 0){
                blockSize = 256;
            } else if ((n_particles % 128) == 0){
                blockSize = 128;
            } else if ((n_particles % 64) == 0){
                blockSize = 64;
            } else if ((n_particles % 32) == 0){
                blockSize = 32;
            }

        }
        int numBlocks = (n_particles + blockSize - 1) / blockSize;

        if (v){
            cout << "numBlocks" << numBlocks << endl;
            cout << "blockSize" << blockSize << endl;
        }
        
        half *d_hmass;
        half2 *d_h2pos;
        float *d_pos, *d_acc_phi, *d_mass, *d_vel;
        float *h_pos = (float*) malloc(n_particles * 3 * sizeof(float));
        float *h_vel = (float*) malloc(n_particles * 3 * sizeof(float));
        float *h_acc_phi = (float*) malloc(n_particles * 4 * sizeof(float));

        cudaMalloc(&d_h2pos,n_particles * 2 * sizeof(half2));
        cudaMalloc(&d_hmass,n_particles * sizeof(half));
        cudaMalloc(&d_acc_phi,n_particles * 4 * sizeof(float));
        cudaMalloc(&d_pos,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_vel,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_mass,n_particles * sizeof(float));

        first = omp_get_wtime();
        cudaMemcpy(d_pos,input_pos,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,input_vel,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,input_mass,n_particles * sizeof(float),cudaMemcpyHostToDevice);

        copyFloat2Half2<<<numBlocks,blockSize>>>(d_pos,d_h2pos);
        copyFloat2Half_dim1<<<numBlocks,blockSize>>>(d_mass,d_hmass);
        second = omp_get_wtime();
        *copyTime += second-first;

        cudaFree(d_mass);

        size_t shared_mem_size = blockSize * 2 * sizeof(half2);

        switch(solver){

            case 0:
                force_solve_cheap<<<numBlocks,blockSize>>>(d_h2pos,d_hmass,d_acc_phi,G,eps,n_particles);
                break;
            case 1:
                force_solve_cheap_shared_mem<<<numBlocks,blockSize,shared_mem_size>>>(d_h2pos,d_hmass,d_acc_phi,G,eps,n_particles);
                break;

        }

        cudaDeviceSynchronize();

        first = omp_get_wtime();
        cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);
        second = omp_get_wtime();
        *copyTime += second-first;

        for (int step = 0; step < steps; step++){

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);
            fast_add_3to3<<<numBlocks,blockSize>>>(d_vel,d_pos,1 * dt);

            copyFloat2Half2<<<numBlocks,blockSize>>>(d_pos,d_h2pos);

            switch(solver){

                case 0:
                    force_solve_cheap<<<numBlocks,blockSize>>>(d_h2pos,d_hmass,d_acc_phi,G,eps,n_particles);
                    break;
                case 1:
                    force_solve_cheap_shared_mem<<<numBlocks,blockSize,shared_mem_size>>>(d_h2pos,d_hmass,d_acc_phi,G,eps,n_particles);
                    break;

            }

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);

            first = omp_get_wtime();
            save(h_pos,h_vel,h_acc_phi,n_particles,fp);
            second = omp_get_wtime();
            *saveTime += second-first;

            cudaDeviceSynchronize();

            first = omp_get_wtime();
            cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);
            second = omp_get_wtime();
            *copyTime += second-first;

        }

        first = omp_get_wtime();
        save(h_pos,h_vel,h_acc_phi,n_particles,fp);
        second = omp_get_wtime();
        *saveTime += second-first;

        free(h_pos);
        free(h_acc_phi);

        cudaFree(d_h2pos);
        cudaFree(d_pos);
        cudaFree(d_acc_phi);
        cudaFree(d_hmass);

        total_second = omp_get_wtime();

        *totalTime = total_second - total_first;
        
    }
}