# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=ascii

import cython
import numpy as np
cimport numpy as np
import pandas as pd
import os

np.import_array()

cdef extern from "eval.cpp":
    void c_evaluate(char* filename, int steps, float G, float eps, float dt, int n_params)

cdef char* input_file = 'input2.csv'

cpdef object evaluate(str input_file, int steps = 0, float G = 1, float eps = 0, float dt = 1/64, int n_params=10, str temp_file = "out.dat"):

    with open(input_file,"r") as f:
        n_particles = len(f.readlines()) - 1

    cdef char* filename = input_file
    c_evaluate(filename,steps,G,eps,dt,n_params)

    step_labels = np.repeat(np.arange(steps+1),n_particles)
    ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

    raw_data = np.fromfile(temp_file,dtype=np.float32,sep="")

    raw_data = raw_data.reshape((steps+1) * n_particles,10)

    step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)
    data = pd.DataFrame(raw_data,columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

    os.remove(temp_file)

    return pd.concat([step_labels,ids,data],axis=1)


#c_evaluate(input_file,2,1,0,1/64,11)


'''
f = "out.dat"
test = np.fromfile(f,dtype=np.float32,sep="")

steps = 2
n_particles = 10

step_labels = np.repeat(np.arange(steps+1),n_particles)
ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

test = test.reshape((steps+1) * n_particles,10)

step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
ids = pd.DataFrame(ids,columns=["id"],dtype=int)
output = pd.DataFrame(test,columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

out = pd.concat([step_labels,ids,output],axis=1)
print(out)
'''
