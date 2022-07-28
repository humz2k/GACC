#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "omp_eval.h"

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

void vecAdd(DATA_TYPE *s, int dim_s, DATA_TYPE *d, int dim_d, DATA_TYPE mul, int n_items){

    for (int i = 0; i < n_items; i++){

        d[i * dim_d] += s[i * dim_s] * mul;
        d[i * dim_d + 1] += s[i * dim_s + 1] * mul;
        d[i * dim_d + 2] += s[i * dim_s + 2] * mul;
        
    }

}

void copy(DATA_TYPE *s, DATA_TYPE *d, int n_items){

    for (int i = 0; i < n_items; i++){

        d[i] = s[i];

    }

}

void force_solver(DATA_TYPE *pos, DATA_TYPE *mass, DATA_TYPE *acc_phi, DATA_TYPE G, DATA_TYPE eps, int n_particles){

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++){

        DATA_TYPE diffx;
        DATA_TYPE diffy;
        DATA_TYPE diffz;

        DATA_TYPE mass_i;
        DATA_TYPE mass_j;

        DATA_TYPE acc_mul;

        DATA_TYPE dist;

        DATA_TYPE ax;
        DATA_TYPE ay;
        DATA_TYPE az;
        DATA_TYPE gpe;

        mass_i = mass[i];

        ax = 0;
        ay = 0;
        az = 0;
        gpe = 0;

        for (int j = 0; j < n_particles; j++){

            if (i != j){

                mass_j = mass[j];

                diffx = pos[j*3] - pos[i*3];
                diffy = pos[j*3 + 1] - pos[i*3 + 1];
                diffz = pos[j*3 + 2] - pos[i*3 + 2];

                dist = sqrt(diffx*diffx + diffy*diffy + diffz*diffz + eps);

                acc_mul = G * mass_j / (dist * dist * dist);
                ax += acc_mul * diffx;
                ay += acc_mul * diffy;
                az += acc_mul * diffz;
                gpe += (-1) * G * mass_j * mass_i / dist;

            }

            acc_phi[i * 4] = ax;
            acc_phi[i * 4 + 1] = ay;
            acc_phi[i * 4 + 2] = az;
            acc_phi[i * 4 + 3] = gpe;

        }
        
    }

}

extern "C" { 
    void omp_evaluate(DATA_TYPE* input_pos, DATA_TYPE* input_vel, DATA_TYPE* input_mass, int n_particles, int steps, DATA_TYPE G, DATA_TYPE eps, DATA_TYPE dt, double *totalTime){

        double total_first,total_second;
        double first,second;

        total_first = omp_get_wtime();

        std::ofstream out;
        out.open( "out.dat", std::ios::out | std::ios::binary);
        std::ofstream &fp = out;

        DATA_TYPE *pos;
        DATA_TYPE *vel;
        DATA_TYPE *acc_phi;

        pos = (DATA_TYPE*) malloc(n_particles * 3 * sizeof(DATA_TYPE));
        vel = (DATA_TYPE*) malloc(n_particles * 3 * sizeof(DATA_TYPE));
        acc_phi = (DATA_TYPE*) malloc(n_particles * 4 * sizeof(DATA_TYPE));

        copy(input_pos,pos,n_particles * 3);
        copy(input_vel,vel,n_particles * 3);

        force_solver(pos,input_mass,acc_phi,G,eps,n_particles);

        save(pos,vel,acc_phi,n_particles,fp);

        for (int step = 0; step < steps; step++){

            vecAdd(acc_phi,4,vel,3,0.5*dt,n_particles*3);
            vecAdd(vel,3,pos,3,1*dt,n_particles*3);

            force_solver(pos,input_mass,acc_phi,G,eps,n_particles);

            vecAdd(acc_phi,4,vel,3,0.5*dt,n_particles*3);

            save(pos,vel,acc_phi,n_particles,fp);

        }

        free(pos);
        free(vel);
        free(acc_phi);

        total_second = omp_get_wtime();

        *totalTime = total_second-total_first;

    }
}