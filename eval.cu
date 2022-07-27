#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "eval.h"

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

__global__
void cuda_parallel(float* pos, float* acc_phi, float G, float eps, int n_particles){

     //TODO: use local memory: make shared memory array of size blocksize. 

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;

    float x_i = pos[i*4];
    float y_i = pos[i*4 + 1];
    float z_i = pos[i*4 + 2];
    float mass_i = pos[i*4 + 3];

    float x_diff;
    float y_diff;
    float z_diff;

    float x_j;
    float y_j;
    float z_j;
    float mass_j;

    float ax = 0;
    float ay = 0;
    float az = 0;
    float gpe = 0;

    float temp0;
    float temp1;
    float temp2;

    float dist;

    for (j = 0; j < n_particles; j++){

        if (i != j){

            x_j = pos[j*4];
            y_j = pos[j*4 + 1];
            z_j = pos[j*4 + 2];
            mass_j = pos[j*4 + 3];

            x_diff = x_j - x_i;
            y_diff = y_j - y_i;
            z_diff = z_j - z_i;

            temp1 = eps;

            temp0 = pow(x_diff,2);
            temp1 += temp0;
            temp0 = pow(y_diff,2);
            temp1 += temp0;
            temp0 = pow(z_diff,2);
            temp1 += temp0;

            dist = sqrt(temp1);

            if (dist != 0){

                temp0 = pow(dist,3);
                temp2 = G * mass_j;
                temp1 = temp2 / temp0;

                temp0 = x_diff * temp1;
                ax += temp0;

                temp0 = y_diff * temp1;
                ay += temp0;

                temp0 = z_diff * temp1;
                az += temp0;

                temp0 = -G;
                temp1 = mass_j * mass_i;
                temp2 = temp0 * temp1;

                gpe += temp2 / dist;
            }

        }
    }

    acc_phi[i*4] = ax;
    acc_phi[i*4 + 1] = ay;
    acc_phi[i*4 + 2] = az;
    acc_phi[i*4 + 3] = gpe;

}

void save(float* pos, float* vel, float* phi_acc, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){

        out.write( reinterpret_cast<const char*>( &pos[i*4 + 0]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &pos[i*4 + 1]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &pos[i*4 + 2]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vel[i*3 + 0]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vel[i*3 + 1]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vel[i*3 + 2]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i*4 + 0]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i*4 + 1]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i*4 + 2]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i*4 + 3]), sizeof( float ));

    }

}

__global__
void fast_add(float *s, float *d, float mul, int sDim0, int dDim0){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d[i * dDim0 + 0] = d[i * dDim0 + 0] + s[i * sDim0 + 0] * mul;
    d[i * dDim0 + 1] = d[i * dDim0 + 1] + s[i * sDim0 + 1] * mul;
    d[i * dDim0 + 2] = d[i * dDim0 + 2] + s[i * sDim0 + 2] * mul;

}

extern "C" { 
    void c_evaluate(float* input_pos, float* input_vel, int n_particles, int steps, float G, float eps, float dt, int n_params){

        std::ofstream out;
        out.open( "out.dat", std::ios::out | std::ios::binary);
        std::ofstream &fp = out;

        int blockSize = 256;
        int numBlocks = (n_particles + blockSize - 1) / blockSize;

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

        cuda_parallel<<<numBlocks,blockSize>>>(d_pos,d_acc_phi,G,eps,n_particles);

        for (int step = 0; step < steps + 1; step++){

            cudaMemcpy(h_pos,d_pos,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * 3 * sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * 4 * sizeof(float),cudaMemcpyDeviceToHost);

            save(h_pos,h_vel,h_acc_phi,n_particles,fp);



        }
        
        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc_phi);
        free(h_pos);
        free(h_vel);
        free(h_acc_phi);

    }

}

int main(int argc, char* argv[]) {

    const int n_params = 10;

    char* filename;
    float eps = 0;
    int steps = 0;
    float G = 1;
    float dt = 1/64;

    if (argc == 1){
        fprintf(stderr,"FILE NOT SPECIFIED\n");
        exit(1);
    }

    filename = argv[1];
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-eps")) eps = atof(argv[i + 1]);
        if (!strcmp(argv[i], "-steps")) steps = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "-G")) G = atof(argv[i + 1]);
        if (!strcmp(argv[i], "-dt")) dt = atof(argv[i + 1]);
    }

    //c_evaluate(filename,steps,G,eps,dt,n_params);

    return 0;
}