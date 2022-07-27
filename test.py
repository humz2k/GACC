# ctypes_test.py
import ctypes
import pathlib
import numpy as np
import pandas as pd
import time

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "eval.dll"
    c_lib = ctypes.CDLL(libname)

def evaluate(pos,vel,mass,steps=0,G=1,eps=0,dt=1/64,n_params=10,outname="out.dat"):

    posPtr = pos.flatten().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    velPtr = vel.flatten().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    massPtr = mass.flatten().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    n_particles = pos.shape[0]

    step_labels = np.repeat(np.arange(steps+1),n_particles)
    ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

    c_lib.c_evaluate(posPtr,velPtr,massPtr,ctypes.c_int(n_particles),ctypes.c_int(steps),ctypes.c_float(G),ctypes.c_float(eps),ctypes.c_float(dt),ctypes.c_int(n_params))

    step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)
    data = pd.DataFrame(np.fromfile(outname,dtype=np.float32,sep="").reshape((steps+1)*n_particles,n_params),columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

    return pd.concat([step_labels,ids,data],axis=1)

a = pd.read_csv("input3.csv")
pos = a.loc[:,["x","y","z"]].to_numpy()
vel = a.loc[:,["vx","vy","vz"]].to_numpy()
mass = a.loc[:,"mass"].to_numpy()

first = time.perf_counter()

b = evaluate(pos,vel,mass,steps=0,G=1,eps=0,dt=1/64)

second = time.perf_counter()

print(a)
print(b)

print(second-first)