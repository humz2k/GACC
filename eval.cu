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

void printFloat3(float3* array, int length){
        for (int i = 0; i < length; i++){
            cout << array[i].x << ",";
            cout << array[i].y << ",";
            cout << array[i].z << endl;
        }
    }

void save(float3* pos, float3* vel, float4* phi_acc, int n_particles, std::ofstream &out){

    for (int i = 0; i < n_particles; i++){

        out.write( reinterpret_cast<const char*>( &pos[i].x), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &pos[i].y), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &pos[i].z), sizeof( float ));

        out.write( reinterpret_cast<const char*>( &vel[i].x), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vel[i].y), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &vel[i].z), sizeof( float ));

        out.write( reinterpret_cast<const char*>( &phi_acc[i].x), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i].y), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i].z), sizeof( float ));
        out.write( reinterpret_cast<const char*>( &phi_acc[i].w), sizeof( float ));

    }

}

void load2float4(float *s, float4 *d, int n){
    for (int i = 0; i < n; i++){

        d[i].x = s[i * 4];
        d[i].y = s[i * 4 + 1];
        d[i].z = s[i * 4 + 2];
        d[i].w = s[i * 4 + 3];

    }
}

void load2float3(float *s, float3 *d, int n){
    for (int i = 0; i < n; i++){

        d[i].x = s[i * 3];
        d[i].y = s[i * 3 + 1];
        d[i].z = s[i * 3 + 2];

    }
}

extern "C" { 
    void c_evaluate(float* input_pos, float* input_vel, float* input_mass, int n_particles, int steps, float G, float eps, float dt, int n_params){

        std::ofstream out;
        out.open( "out.dat", std::ios::out | std::ios::binary);
        std::ofstream &fp = out;

        int blockSize = 256;
        int numBlocks = (n_particles + blockSize - 1) / blockSize;

        float3 *d_pos, *d_vel, *h_pos, *h_vel;
        float4 *d_acc_phi, *h_acc_phi;
        float *d_mass;

        h_pos = (float3*) malloc(n_particles * sizeof(float3));
        h_acc_phi = (float4*) malloc(n_particles * sizeof(float4));
        h_vel = (float3*) malloc(n_particles * sizeof(float3));

        load2float3(input_pos,h_pos,n_particles);
        load2float3(input_vel,h_vel,n_particles);

        cudaMalloc(&d_pos,n_particles * sizeof(float3));
        cudaMalloc(&d_acc_phi,n_particles * sizeof(float4));
        cudaMalloc(&d_vel,n_particles * sizeof(float3));
        cudaMalloc(&d_mass,n_particles * sizeof(float));

        cudaMemcpy(d_pos,h_pos,n_particles * sizeof(float3),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,h_vel,n_particles * sizeof(float3),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,input_mass,n_particles * sizeof(float),cudaMemcpyHostToDevice);

        //printFloat3(h_pos,n_particles);
        //cout << "" << endl;
        //printFloat3(h_vel,n_particles);
        //cout << "" << endl;
        //printArray(input_mass,n_particles);

        force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);

        for (int step = 0; step < steps + 1; step++){

            cudaMemcpy(h_pos,d_pos,n_particles * sizeof(float3),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel,d_vel,n_particles * sizeof(float3),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_acc_phi,d_acc_phi,n_particles * sizeof(float4),cudaMemcpyDeviceToHost);

            printFloat3(h_vel,n_particles);

            save(h_pos,h_vel,h_acc_phi,n_particles,fp);

            //fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5);
            //fast_add_3to3<<<numBlocks,blockSize>>>(d_vel,d_pos,1);

            //force_solve_gpu<<<numBlocks,blockSize>>>(d_pos,d_mass,d_acc_phi,G,eps,n_particles);

            //fast_add_4to3<<<numBlocks,blockSize>>>(d_acc_phi,d_vel,0.5);

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