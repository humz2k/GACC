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

void printArray(float* array, int length){
    for (int i = 0; i < length; i++){
        printf("%lf\n",array[i]);
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
void fast_add(float* sx, float* sy, float* sz, float* dx, float* dy, float* dz, float mul){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    dx[i] = dx[i] + sx[i] * mul;
    dy[i] = dy[i] + sy[i] * mul;
    dz[i] = dz[i] + sz[i] * mul;

}

extern "C" { 
    void c_evaluate(char* filename, int steps, float G, float eps, float dt, int n_params){

    std::ofstream out;
    out.open( "out.dat", std::ios::out | std::ios::binary);

    std::ofstream &fp = out;

    //get number of particles
    int n_particles = countLines(filename)-1;

    int blockSize = 256;
    int numBlocks = (n_particles + blockSize - 1) / blockSize;

    size_t size = n_particles * sizeof(float);

    float *x,*y,*z,*vx,*vy,*vz,*ax,*ay,*az,*gpe,*mass;
    
    cudaMalloc(&x, size);
    cudaMalloc(&y, size);
    cudaMalloc(&z, size);
    cudaMalloc(&vx, size);
    cudaMalloc(&vy, size);
    cudaMalloc(&vz, size);
    cudaMalloc(&ax, size);
    cudaMalloc(&ay, size);
    cudaMalloc(&az, size);
    cudaMalloc(&gpe, size);
    cudaMalloc(&mass, size);

    float *copy_x = (float*) malloc(size);
    float *copy_y = (float*) malloc(size);
    float *copy_z = (float*) malloc(size);
    float *copy_vx = (float*) malloc(size);
    float *copy_vy = (float*) malloc(size);
    float *copy_vz = (float*) malloc(size);
    float *copy_ax = (float*) malloc(size);
    float *copy_ay = (float*) malloc(size);
    float *copy_az = (float*) malloc(size);
    float *copy_gpe = (float*) malloc(size);
    float *copy_mass = (float*) malloc(size);
    
    readFile(filename,copy_x,copy_y,copy_z,copy_vx,copy_vy,copy_vz,copy_mass,",");

    cudaMemcpy(x, copy_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y, copy_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(z, copy_z, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vx, copy_vx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vy, copy_vy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vz, copy_vz, size, cudaMemcpyHostToDevice);
    cudaMemcpy(ax, copy_ax, size, cudaMemcpyHostToDevice);
    cudaMemcpy(ay, copy_ay, size, cudaMemcpyHostToDevice);
    cudaMemcpy(az, copy_az, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpe, copy_gpe, size, cudaMemcpyHostToDevice);
    cudaMemcpy(mass, copy_mass, size, cudaMemcpyHostToDevice);

    float eps_square = pow(eps,2);


    cuda_parallel<<<numBlocks, blockSize>>>(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);

    for (int step = 0; step < steps + 1; step++){

        cudaDeviceSynchronize();

        cudaMemcpy(copy_x, x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_y, y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_z, z, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_vx, vx, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_vy, vy, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_vz, vz, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_ax, ax, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_ay, ay, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_az, az, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(copy_gpe, gpe, size, cudaMemcpyDeviceToHost);

        save(step,copy_x,copy_y,copy_z,copy_vx,copy_vy,copy_vz,copy_ax,copy_ay,copy_az,copy_gpe,n_particles,fp);

        fast_add<<<numBlocks,blockSize>>>(ax,ay,az,vx,vy,vz,0.5);
        fast_add<<<numBlocks,blockSize>>>(vx,vy,vz,x,y,z,1);

        cuda_parallel<<<numBlocks, blockSize>>>(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);

        fast_add<<<numBlocks,blockSize>>>(ax,ay,az,vx,vy,vz,0.5);
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(vx);
    cudaFree(vy);
    cudaFree(vz);
    cudaFree(ax);
    cudaFree(ay);
    cudaFree(az);
    cudaFree(gpe);
    cudaFree(mass);

    
    free(copy_x);
    free(copy_y);
    free(copy_z);
    free(copy_vx);
    free(copy_vy);
    free(copy_vz);
    free(copy_ax);
    free(copy_ay);
    free(copy_az);
    free(copy_gpe);
    free(copy_mass);

    out.close();

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

    c_evaluate(filename,steps,G,eps,dt,n_params);

    return 0;
}