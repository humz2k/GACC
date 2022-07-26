# ctypes_test.py
import ctypes
import pathlib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "eval.dll"
    c_lib = ctypes.CDLL(libname)

def evaluate(pos,vel,steps=0,G=1,eps=0,dt=1/64,n_params=10):

    posPtr = pos.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    velPtr = vel.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    n_particles = ctypes.c_int(pos.shape[0])

    return c_lib.c_evaluate(posPtr,velPtr,n_particles,ctypes.c_int(steps),ctypes.c_float(G),ctypes.c_float(eps),ctypes.c_float(dt),ctypes.c_int(n_params))

a = pd.read_csv("input.csv")
pos = a.loc[:,["x","y","z","mass"]].to_numpy()
vel = a.loc[:,["vx","vy","vz"]].to_numpy()

evaluate(pos,vel)