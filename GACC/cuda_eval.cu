#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "cuda_eval.h"
#include "kernels/kernels.h"

using namespace std;

void save(DATA_TYPE* pos, DATA_TYPE* vel, DATA_TYPE* phi_acc, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){

        out.write( reinterpret_cast<char*>( &pos[i*3]), sizeof( DATA_TYPE ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 1]), sizeof( DATA_TYPE ));
        out.write( reinterpret_cast<char*>( &pos[i*3 + 2]), sizeof( DATA_TYPE ));

        out.write( reinterpret_cast<char*>( &vel[i*3]), sizeof( DATA_TYPE ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 1]), sizeof( DATA_TYPE ));
        out.write( reinterpret_cast<char*>( &vel[i*3 + 2]), sizeof( DATA_TYPE ));

        out.write( reinterpret_cast<char*>( &phi_acc[i*4]), sizeof( DATA_TYPE ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 1]), sizeof( DATA_TYPE ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 2]), sizeof( DATA_TYPE ));
        out.write( reinterpret_cast<char*>( &phi_acc[i*4 + 3]), sizeof( DATA_TYPE ));

    }

}

extern "C" { 
    void cuda_evaluate(DATA_TYPE* input_pos, DATA_TYPE* input_vel, DATA_TYPE* input_mass, int n_particles, int steps, DATA_TYPE G, DATA_TYPE eps, DATA_TYPE dt, int n_params, int solver, int v, double *saveTime, double *totalTime, double *copyTime){

        double first,second;
        double total_first,total_second;

        cudaFree(0);

        *saveTime = 0;
        *totalTime = 0;
        *copyTime = 0;

        total_first = omp_get_wtime();

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

        DATA_TYPE *h_pos = (DATA_TYPE*) malloc(n_particles * 3 * sizeof(DATA_TYPE));
        DATA_TYPE *h_acc_phi = (DATA_TYPE*) malloc(n_particles * 4 * sizeof(DATA_TYPE));
        DATA_TYPE *h_vel = (DATA_TYPE*) malloc(n_particles * 3 * sizeof(DATA_TYPE));

        DATA_TYPE *d_pos;
        cudaMalloc(&d_pos,n_particles * 3 * sizeof(DATA_TYPE));

        DATA_TYPE *d_acc_phi;
        cudaMalloc(&d_acc_phi,n_particles * 4 * sizeof(DATA_TYPE));

        DATA_TYPE *d_vel;
        cudaMalloc(&d_vel,n_particles * 3 * sizeof(DATA_TYPE));

        DATA_TYPE *d_mass;
        cudaMalloc(&d_mass,n_particles * sizeof(DATA_TYPE));

        first = omp_get_wtime();
        cudaMemcpy(d_pos,input_pos,n_particles * 3 * sizeof(DATA_TYPE),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,input_vel,n_particles * 3 * sizeof(DATA_TYPE),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,input_mass,n_particles * sizeof(DATA_TYPE),cudaMemcpyHostToDevice);
        second = omp_get_wtime();
        *copyTime += second-first;

        size_t shared_mem_size = blockSize * 4 * sizeof(DATA_TYPE);

        switch (solver){

            case 0:
                force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                break;
            
            case 1:
                force_solve_shared_mem<<<numBlocks,blockSize,shared_mem_size>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                break;

        }

        cudaDeviceSynchronize();

        first = omp_get_wtime();
        cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(DATA_TYPE),cudaMemcpyDeviceToHost);
        second = omp_get_wtime();
        *copyTime += second-first;

        *saveTime += second-first;


        for (int step = 0; step < steps; step++){

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);
            fast_add_3to3<<<numBlocks,blockSize>>>(d_vel,d_pos,1 * dt);

            switch (solver){

                case 0:
                    force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                    break;
                
                case 1:
                    force_solve_shared_mem<<<numBlocks,blockSize,shared_mem_size>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                    break;

            }

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);

            first = omp_get_wtime();
            save(h_pos,h_vel,h_acc_phi,n_particles,fp);
            second = omp_get_wtime();
            *saveTime += second-first;

            cudaDeviceSynchronize();

            first = omp_get_wtime();
            cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(DATA_TYPE),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(DATA_TYPE),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(DATA_TYPE),cudaMemcpyDeviceToHost);
            second = omp_get_wtime();
            *copyTime += second-first;

        }

        first = omp_get_wtime();
        save(h_pos,h_vel,h_acc_phi,n_particles,fp);
        second = omp_get_wtime();
        *saveTime += second-first;

        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc_phi);
        cudaFree(d_mass);

        free(h_pos);
        free(h_vel);
        free(h_acc_phi);

        total_second = omp_get_wtime();

        *totalTime = total_second - total_first;
    }
}