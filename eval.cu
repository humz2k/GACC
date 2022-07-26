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
void cuda_parallel(float* x, float* y, float* z, float* ax, float* ay, float* az, float* gpe, float* mass, float G, float eps, int n_particles){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    //cudaMemset

    //use local memory: make shared memory array of size blocksize. 

    ax[i] = 0;
    ay[i] = 0;
    az[i] = 0;
    gpe[i] = 0;

    for (int j = 0; j < n_particles; j++){

        if (i != j){

            float dist = sqrt(pow((x[i] - x[j]),2) + pow((y[i] - y[j]),2) + pow((z[i] - z[j]),2) + eps);
            float acc_mul = G * mass[j] / pow(dist,3);

            ax[i] = ax[i] + (x[j] - x[i]) * acc_mul;
            ay[i] = ay[i] + (y[j] - y[i]) * acc_mul;
            az[i] = az[i] + (z[j] - z[i]) * acc_mul;
            gpe[i] = gpe[i] + (-1) * G * mass[j] * mass[i] / dist;

        }
    }

}

void save(int step, float* x, float* y, float* z, float* vx, float* vy, float* vz, float* ax, float* ay, float* az, float* gpe, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){
        
        out.write( reinterpret_cast<const char*>( &x[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &y[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &z[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vx[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vy[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vz[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &ax[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &ay[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &az[i]), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &gpe[i]), sizeof( float ));

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

        int blockSize = 256;
        int numBlocks = (n_particles + blockSize - 1) / blockSize;

        int pos_width = 4, vel_width = 3, height = n_particles;
        float *d_pos,*d_vel,*d_acc_phi;
        float *h_pos,*h_vel,*h_acc_phi;
        size_t pos_input_pitch = 4 * sizeof(float), vel_input_pitch = 3 * sizeof(float);

        size_t pos_pitch;
        cudaMallocPitch(&d_pos, &pos_pitch, pos_width * sizeof(float), height);
        cudaMemcpy2D(d_pos,pos_pitch,input_pos,pos_input_pitch,pos_input_pitch,n_particles,cudaMemcpyHostToDevice);
        h_pos = (float*) malloc(n_particles * 4 * sizeof(float));

        size_t vel_pitch;
        cudaMallocPitch(&d_vel, &vel_pitch, vel_width * sizeof(float), height);
        cudaMemcpy2D(d_pos,vel_pitch,input_pos,vel_input_pitch,vel_input_pitch,n_particles,cudaMemcpyHostToDevice);
        h_vel = (float*) malloc(n_particles * 3 * sizeof(float));

        size_t acc_pitch;
        cudaMallocPitch(&d_acc_phi, &acc_pitch, pos_width * sizeof(float), height);
        h_acc_phi = (float*) malloc(n_particles * 4 * sizeof(float));

        fast_add<<<numBlocks,blockSize>>>(d_vel,d_pos,1,3,4);
        cudaMemcpy2D(h_pos,pos_input_pitch,d_pos,pos_pitch,pos_pitch,n_particles,cudaMemcpyDeviceToHost);

        //SAVE THIS TO FILE AND READ IN PYTHON TO SEE IF IT WORKS

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