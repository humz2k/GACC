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
    void c_evaluate(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, int n_params){

        std::ofstream out;
        out.open( "out.dat", std::ios::out | std::ios::binary);
        std::ofstream &fp = out;

        int blockSize = 256;
        int numBlocks = (n_particles + blockSize - 1) / blockSize;

        float *d_pos, *d_vel, *d_acc_phi, *h_pos, *h_vel, *h_acc_phi, *d_mass;

        h_pos = (float*) malloc(n_particles * 3 * sizeof(float));
        h_acc_phi = (float*) malloc(n_particles * 4 * sizeof(float));
        h_vel = (float*) malloc(n_particles * 3 * sizeof(float));

        cudaMalloc(&d_pos,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_acc_phi,n_particles * 4 * sizeof(float));
        cudaMalloc(&d_vel,n_particles * 3 * sizeof(float));
        cudaMalloc(&d_mass,n_particles * sizeof(float));

        cudaMemcpy(d_pos,input_pos,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,input_vel,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,input_mass,n_particles * sizeof(float),cudaMemcpyHostToDevice);

        force_solve_gpu<<<numBlocks,blockSize,n_particles*4*sizeof(float)>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);

        for (int step = 0; step < steps + 1; step++){

            cudaMemcpy(h_pos,d_pos,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);

            save(h_pos,h_vel,h_acc_phi,n_particles,fp);

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5);
            fast_add_3to3<<<numBlocks,blockSize>>>(d_vel,d_pos,1);

            force_solve_gpu<<<numBlocks,blockSize,n_particles*4*sizeof(float)>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);

            fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5);

        }

        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc_phi);
        cudaFree(d_mass);

        free(h_pos);
        free(h_vel);
        free(h_acc_phi);


        /*

        int pos_width = 4, vel_width = 3, height = n_particles;

        float *d_pos,*d_vel,*d_acc_phi;
        float *h_pos,*h_vel,*h_acc_phi;

        size_t size4 = (n_particles * 4) * sizeof(float);
        size_t size3 = (n_particles * 3) * sizeof(float);
        
        h_pos = (float*) malloc(size4);
        cudaMalloc(&d_pos,size4);
        cudaMemcpy(d_pos,input_pos,n_particles * 4 * sizeof(float),cudaMemcpyHostToDevice);

        h_vel = (float*) malloc(size3);
        cudaMalloc(&d_vel,size3);
        cudaMemcpy(d_vel,input_vel,n_particles * 3 * sizeof(float),cudaMemcpyHostToDevice);

        h_acc_phi = (float*) malloc(size4);
        cudaMalloc(&d_acc_phi,size4);

        force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_acc_phi,G,eps,n_particles);

        for (int step = 0; step < steps + 1; step++){

            cudaMemcpy(h_pos,d_pos,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);

            save(h_pos,h_vel,h_acc_phi,n_particles,fp);

            fast_add<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5,4,3);
            fast_add<<<numBlocks,blockSize>>>(d_vel,d_pos,1,3,4);

            force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_acc_phi,G,eps,n_particles);

            fast_add<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5,4,3);

        }
        
        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc_phi);
        free(h_pos);
        free(h_vel);
        free(h_acc_phi);
        */
    }
}

int main(){

    return 0;
}