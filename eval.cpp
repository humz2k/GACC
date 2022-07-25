#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>

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

void phi_acc(float* x, float* y, float* z, float* ax, float* ay, float* az, float* gpe, float* mass, float G, int n_particles){

    for (int i = 0; i < n_particles; i++){

        ax[i] = 0;
        ay[i] = 0;
        az[i] = 0;
        gpe[i] = 0;

        for (int j = 0; j < n_particles; j++){

            if (i != j){

                float dist = sqrt(pow((x[i] - x[j]),2) + pow((y[i] - y[j]),2) + pow((z[i] - z[j]),2));
                float acc_mul = G * mass[j] / pow(dist,3);

                ax[i] = ax[i] + (x[j] - x[i]) * acc_mul;
                ay[i] = ay[i] + (y[j] - y[i]) * acc_mul;
                az[i] = az[i] + (z[j] - z[i]) * acc_mul;
                gpe[i] = gpe[i] + (-1) * G * mass[j] * mass[i] / dist;

            }
        }
    }
}

void save(int step, float* x, float* y, float* z, float* vx, float* vy, float* vz, float* ax, float* ay, float* az, float* gpe, int n_particles, float* output, int n_params){

    int index = step * n_params * n_particles;

    for (int i = 0; i < n_particles; i++){
        
        output[index + 0] = x[i];
        output[index + 1] = y[i];
        output[index + 2] = z[i];
        output[index + 3] = vx[i];
        output[index + 4] = vy[i];
        output[index + 5] = vz[i];
        output[index + 6] = ax[i];
        output[index + 7] = ay[i];
        output[index + 8] = az[i];
        output[index + 9] = gpe[i];

        index = index + n_params;
    }
    

}

void add(float* source, float* dest, float mul, int len){

    for (int i = 0; i < len; i++){

        dest[i] = dest[i] + source[i] * mul;

    }

}

void evaluate(char* filename, int steps, float G, float eps, float dt, int n_params){

    std::ofstream out;
    out.open( "out.dat", std::ios::out | std::ios::binary);


    //get number of particles
    int n_particles = countLines(filename)-1;

    //malloc arrays
    float* x = (float*) malloc(n_particles * sizeof(float));
    float* y = (float*) malloc(n_particles * sizeof(float));
    float* z = (float*) malloc(n_particles * sizeof(float));
    float* vx = (float*) malloc(n_particles * sizeof(float));
    float* vy = (float*) malloc(n_particles * sizeof(float));
    float* vz = (float*) malloc(n_particles * sizeof(float));
    float* ax = (float*) malloc(n_particles * sizeof(float));
    float* ay = (float*) malloc(n_particles * sizeof(float));
    float* az = (float*) malloc(n_particles * sizeof(float));
    float* gpe = (float*) malloc(n_particles * sizeof(float));
    float* mass = (float*) malloc(n_particles * sizeof(float));

    //malloc arrays to save
    float* output = (float*) malloc((steps+1) * n_particles * n_params * sizeof(float));
    
    readFile(filename,x,y,z,vx,vy,vz,mass,",");

    phi_acc(x,y,z,ax,ay,az,gpe,mass,G,n_particles);

    for (int step = 0; step < steps + 1; step++){
        save(step,x,y,z,vx,vy,vz,ax,ay,az,gpe,n_particles,output,n_params);

        add(ax,vx,0.5,n_particles);
        add(ay,vy,0.5,n_particles);
        add(az,vz,0.5,n_particles);

        add(vx,x,1,n_particles);
        add(vy,y,1,n_particles);
        add(vz,z,1,n_particles);

        phi_acc(x,y,z,ax,ay,az,gpe,mass,G,n_particles);

        add(ax,vx,0.5,n_particles);
        add(ay,vy,0.5,n_particles);
        add(az,vz,0.5,n_particles);
    }

    for (int i = 0; i < (steps+1) * n_particles * n_params; i++){
        out.write( reinterpret_cast<const char*>( &output[i]), sizeof( float ));
    }

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
    free(output);

    out.close();

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

    evaluate(filename,steps,G,eps,dt,n_params);

    return 0;
}