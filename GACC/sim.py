# ctypes_test.py
import ctypes
import pathlib
import numpy as np
import pandas as pd
import time
from math import ceil
import os

libname = pathlib.Path().absolute() / "eval.dll"
c_lib = ctypes.CDLL(libname)

def pad(df):
    pos = df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float32)
    vel = df.loc[:,["vx","vy","vz"]].to_numpy().flatten().astype(np.float32)
    mass = df.loc[:,"mass"].to_numpy().astype(np.float32)
    max_x = ceil(np.max(pos)) + 1
    n_particles = len(df.index)
    optimal_n = ((n_particles // 128) + ceil((n_particles % 128)/128)) * 128
    diff = optimal_n - n_particles
    new_masses = np.zeros(diff,dtype=np.float32)
    new_vels = np.zeros(diff*3,dtype=np.float32)
    new_pos = np.arange(diff*3,dtype=np.float32)/diff + max_x
    pos = np.concatenate([pos,new_pos])
    vel = np.concatenate([vel,new_vels])
    mass = np.concatenate([mass,new_masses])

    return optimal_n,pos,vel,mass


def evaluate(df,steps=0,G=1,eps=0,dt=1/64,n_params=10, solver = 0,outname="out.dat",v=False):

    true_n = len(df.index)

    n_particles,pos,vel,mass = pad(df)

    pointer = ctypes.POINTER(ctypes.c_double)

    save_time = ctypes.c_double(0)
    total_time = ctypes.c_double(0)
    copy_time = ctypes.c_double(0)
    
    saveTimePtr = pointer(save_time)
    totalTimePtr = pointer(total_time)
    copyTimePtr = pointer(copy_time)

    posPtr = pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    velPtr = vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    massPtr = mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    step_labels = np.repeat(np.arange(steps+1),n_particles)
    ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

    c_lib.c_evaluate(posPtr,velPtr,massPtr,ctypes.c_int(n_particles),ctypes.c_int(steps),ctypes.c_float(G),ctypes.c_float(eps),ctypes.c_float(dt),ctypes.c_int(n_params), ctypes.c_int(solver), ctypes.c_int(v), saveTimePtr, totalTimePtr, copyTimePtr)

    step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)
    data = pd.DataFrame(np.fromfile(outname,dtype=np.float32,sep="").reshape((steps+1)*n_particles,n_params),columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

    out_df = pd.concat([step_labels,ids,data],axis=1)

    stats = {}
    stats["total_save_time"] = save_time.value
    stats["total_copy_time"] = copy_time.value
    stats["total_eval_time"] = total_time.value

    os.remove('out.dat')

    return out_df[(out_df["id"] < true_n)].reset_index(),stats