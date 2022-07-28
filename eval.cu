#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "eval.h"
#include "kernels/kernels.h"

using namespace std;

int countLines(char* filename){
    FILE* filePointer;
    filePointer = fopen(filename, "r");
    int length = 0;
    if (!(filePointer == NULL)) {
        char buffer[256];
        while (fgets(buffer,256,filePointer)){
            length++;
        }
    }
    fclose(filePointer);
    return length;
}

int readFile(char* filename,float* x,float* y,float* z,float* vx,float* vy,float* vz,float* mass, const char* delim){

    FILE* filePointer;
    filePointer = fopen(filename, "r");

    int count = -1;

    if (!(filePointer == NULL)) {

        char buffer[256];
        count = -1;
        int index;

        while (fgets(buffer,256,filePointer)){
            if (count != -1){
                index = 0;

                char *line = strtok(buffer, delim);
                while (line != NULL){

                    switch(index){

                        case 1:
                            x[count] = atof(line);
                            break;
                        case 2:
                            y[count] = atof(line);
                            break;
                        case 3:
                            z[count] = atof(line);
                            break;
                        case 4:
                            vx[count] = atof(line);
                            break;
                        case 5:
                            vy[count] = atof(line);
                            break;
                        case 6:
                            vz[count] = atof(line);
                            break;
                        case 7:
                            mass[count] = atof(line);
                            break;

                    }
                    
                    index++;
                    line = strtok(NULL, delim);
                }
            }
        count++;
        }
    }
    
    fclose(filePointer);

    return count + 1;

}

extern "C" {
    void printArray(float* array, int length){
        for (int i = 0; i < length; i++){
            printf("%lf\n",array[i]);
        }
    }
}

void save(float* pos, float* vel, float* phi_acc, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){

        out.write( reinterpret_cast<const char*>( &pos[i*3]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &pos[i*3 + 1]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &pos[i*3 + 2]), sizeof( float ));

        out.write( reinterpret_cast<const char*>( &vel[i*3]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vel[i*3 + 1]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vel[i*3 + 2]), sizeof( float ));

        out.write( reinterpret_cast<const char*>( &phi_acc[i*4]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i*4 + 1]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i*4 + 2]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i*4 + 3]), sizeof( float ));

    }

}

extern "C" { 
    void c_evaluate(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, int n_params, int solver, int v, double *saveTime, double *totalTime, double *copyTime){

        double first,second;
        double total_first,total_second;

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

        float *h_pos = (float*) malloc(n_particles * 3 * sizeof(float));
        float *h_acc_phi = (float*) malloc(n_particles * 4 * sizeof(float));
        float* h_vel = (float*) malloc(n_particles * 3 * sizeof(float));

        float *d_pos;
        cudaMalloc(&d_pos,n_particles * 3 * sizeof(float));

        float *d_acc_phi;
        cudaMalloc(&d_acc_phi,n_particles * 4 * sizeof(float));

        float *d_vel;
        cudaMalloc(&d_vel,n_particles * 3 * sizeof(float));

        float *d_mass;
        cudaMalloc(&d_mass,n_particles * sizeof(float));

        first = omp_get_wtime();
        cudaMemcpy(d_pos,input_pos,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,input_vel,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,input_mass,n_particles * sizeof(float),cudaMemcpyHostToDevice);
        second = omp_get_wtime();
        *copyTime += second-first;


        switch (solver){

            case 0:
                force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                break;
            
            case 1:
                force_solve_shared_mem<<<numBlocks,blockSize,blockSize * 4 * sizeof(float)>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                break;

        }

        cudaDeviceSynchronize();

        //cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
        first = omp_get_wtime();
        cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);
        second = omp_get_wtime();
        *copyTime += second-first;

        first = omp_get_wtime();
        save(input_pos,input_vel,h_acc_phi,n_particles,fp);
        second = omp_get_wtime();

        *saveTime += second-first;


        for (int step = 0; step < steps; step++){

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);
            fast_add_3to3<<<numBlocks,blockSize,n_particles * 4 * sizeof(float)>>>(d_vel,d_pos,1 * dt);

            switch (solver){

                case 0:
                    force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                    break;
                
                case 1:
                    force_solve_shared_mem<<<numBlocks,blockSize,blockSize * 4 * sizeof(float)>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);
                    break;

            }

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5 * dt);

            cudaDeviceSynchronize();

            first = omp_get_wtime();
            cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);
            second = omp_get_wtime();
            *copyTime += second-first;

            first = omp_get_wtime();
            save(h_pos,h_vel,h_acc_phi,n_particles,fp);
            second = omp_get_wtime();
            *saveTime += second-first;

        }

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

int main(){

    return 0;
}