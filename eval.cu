#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>

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

void no_parallel(float* x, float* y, float* z, float* ax, float* ay, float* az, float* gpe, float* mass, float G, float eps, int n_particles){

    for (int i = 0; i < n_particles; i++){

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
}


void cpu_parallel(float* x, float* y, float* z, float* ax, float* ay, float* az, float* gpe, float* mass, float G, float eps, int n_particles){

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++){

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
}

__global__
void cuda_parallel(float* x, float* y, float* z, float* ax, float* ay, float* az, float* gpe, float* mass, float G, float eps, int n_particles){

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n_particles; i += stride){

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

void add(float* source, float* dest, float mul, int len){

    for (int i = 0; i < len; i++){

        dest[i] = dest[i] + source[i] * mul;

    }

}

__global__
void fast_add(float* sx, float* sy, float* sz, float* dx, float* dy, float* dz, float mul, int len){

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < len; i += stride){

        dx[i] = dx[i] + sx[i] * mul;
        dy[i] = dy[i] + sy[i] * mul;
        dz[i] = dz[i] + sz[i] * mul;

    }

}

void c_evaluate(char* filename, int steps, float G, float eps, float dt, int n_params, int backend){

    std::ofstream out;
    out.open( "out.dat", std::ios::out | std::ios::binary);

    std::ofstream &fp = out;

    //get number of particles
    int n_particles = countLines(filename)-1;

    int blockSize = 256;
    int numBlocks = (n_particles + blockSize - 1) / blockSize;

    float *x,*y,*z,*vx,*vy,*vz,*ax,*ay,*az,*gpe,*mass;

    if (backend < 2){

    //malloc arrays
        x = (float*) malloc(n_particles * sizeof(float));
        y = (float*) malloc(n_particles * sizeof(float));
        z = (float*) malloc(n_particles * sizeof(float));
        ax = (float*) malloc(n_particles * sizeof(float));
        ay = (float*) malloc(n_particles * sizeof(float));
        az = (float*) malloc(n_particles * sizeof(float));
        vx = (float*) malloc(n_particles * sizeof(float));
        vy = (float*) malloc(n_particles * sizeof(float));
        vz = (float*) malloc(n_particles * sizeof(float));
        gpe = (float*) malloc(n_particles * sizeof(float));
        mass = (float*) malloc(n_particles * sizeof(float));
    } else{

        cudaMallocManaged(&x, n_particles*sizeof(float));
        cudaMallocManaged(&y, n_particles*sizeof(float));
        cudaMallocManaged(&z, n_particles*sizeof(float));
        cudaMallocManaged(&vx, n_particles*sizeof(float));
        cudaMallocManaged(&vy, n_particles*sizeof(float));
        cudaMallocManaged(&vz, n_particles*sizeof(float));
        cudaMallocManaged(&ax, n_particles*sizeof(float));
        cudaMallocManaged(&ay, n_particles*sizeof(float));
        cudaMallocManaged(&az, n_particles*sizeof(float));
        cudaMallocManaged(&gpe, n_particles*sizeof(float));
        cudaMallocManaged(&mass, n_particles*sizeof(float));
    }
    
    readFile(filename,x,y,z,vx,vy,vz,mass,",");

    float eps_square = pow(eps,2);

    switch(backend){

        case 0:
            no_parallel(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);
            break;
        case 1:
            cpu_parallel(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);
            break;
        case 2:
            cuda_parallel<<<numBlocks, blockSize>>>(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);
            cudaDeviceSynchronize();
            break;

    }

    for (int step = 0; step < steps + 1; step++){
        save(step,x,y,z,vx,vy,vz,ax,ay,az,gpe,n_particles,fp);

        if (backend < 2){
            add(ax,vx,0.5,n_particles);
            add(ay,vy,0.5,n_particles);
            add(az,vz,0.5,n_particles);

            add(vx,x,1,n_particles);
            add(vy,y,1,n_particles);
            add(vz,z,1,n_particles);
        } else {

            fast_add<<<numBlocks,blockSize>>>(ax,ay,az,vx,vy,vz,0.5,n_particles);
            cudaDeviceSynchronize();
            fast_add<<<numBlocks,blockSize>>>(vx,vy,vz,x,y,z,1,n_particles);
            cudaDeviceSynchronize();

        }

        switch(backend){

            case 0:
                no_parallel(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);
                break;
            case 1:
                cpu_parallel(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);
                break;
            case 2:
                cuda_parallel<<<numBlocks, blockSize>>>(x,y,z,ax,ay,az,gpe,mass,G,eps_square,n_particles);
                cudaDeviceSynchronize();
                break;
            
        }

        if (backend < 2){
            add(ax,vx,0.5,n_particles);
            add(ay,vy,0.5,n_particles);
            add(az,vz,0.5,n_particles);
        } else {
            fast_add<<<numBlocks,blockSize>>>(ax,ay,az,vx,vy,vz,0.5,n_particles);
            cudaDeviceSynchronize();
        }
    }

    if (backend < 2){
        free(x);
        free(y);
        free(z);
        free(vx);
        free(vy);
        free(vz);
        free(ax);
        free(az);
        free(ay);
        free(gpe);
        free(mass);
    } else {
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
    }

    out.close();

}

int main(int argc, char* argv[]) {

    const int n_params = 10;

    char* filename;
    float eps = 0;
    int steps = 0;
    float G = 1;
    float dt = 1/64;
    int backend = 0;

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
        if (!strcmp(argv[i], "-backend")) backend = atoi(argv[i + 1]);
    }

    c_evaluate(filename,steps,G,eps,dt,n_params,backend);

    return 0;
}