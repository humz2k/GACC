from subprocess import call
import numpy as np
import pandas as pd
import os
import time

def evaluate(filename,eps=0,G=1,steps=0,dt=1/64):

    backends = {"c++":"0","openmp":"1","cuda":"2"}

    call(["./eval",filename,"-eps",str(eps),"-G",str(G),"-steps", str(steps),"-dt", str(dt)])

    '''
    with open(filename,"r") as f:
        n_particles = len(f.readlines()) - 1

    temp_file = "out.dat"

    step_labels = np.repeat(np.arange(steps+1),n_particles)
    ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

    raw_data = np.fromfile(temp_file,dtype=np.float32,sep="")

    raw_data = raw_data.reshape((steps+1) * n_particles,10)

    step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)
    data = pd.DataFrame(raw_data,columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

    os.remove(temp_file)

    return pd.concat([step_labels,ids,data],axis=1)
    '''

#first = time.perf_counter()
#a = evaluate("input.csv",backend="openmp")
#second = time.perf_counter()
#print(a)

first = time.perf_counter()
a = evaluate("input.csv")
second = time.perf_counter()
print(second-first)
print(a)