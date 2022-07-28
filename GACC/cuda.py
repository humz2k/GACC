# ctypes_test.py
import ctypes
import pathlib
import numpy as np
import pandas as pd
import time
from math import ceil
import os

def pad(df,new_type):
    pos = df.loc[:,["x","y","z"]].to_numpy().flatten()
    vel = df.loc[:,["vx","vy","vz"]].to_numpy().flatten()
    mass = df.loc[:,"mass"].to_numpy()
    max_x = ceil(np.max(pos)) + 1
    n_particles = len(df.index)
    optimal_n = ((n_particles // 128) + ceil((n_particles % 128)/128)) * 128
    diff = optimal_n - n_particles
    new_masses = np.zeros(diff,dtype=float)
    new_vels = np.zeros(diff*3,dtype=float)
    new_pos = np.arange(diff*3,dtype=float)/diff + max_x
    pos = np.concatenate([pos,new_pos])
    vel = np.concatenate([vel,new_vels])
    mass = np.concatenate([mass,new_masses])

    return optimal_n,pos.astype(new_type),vel.astype(new_type),mass.astype(new_type)

package_directory = os.path.dirname(os.path.abspath(__file__))

f4_lib = package_directory + "/cuda_eval_f4.dll"
cuda_eval_f4 = ctypes.CDLL(f4_lib)

f8_lib = package_directory + "/cuda_eval_f8.dll"
cuda_eval_f8 = ctypes.CDLL(f8_lib)

def evaluate(df,steps=0,G=1,eps=0,dt=1/64,n_params=10, solver = 0,outname="out.dat",v=False,precision="f4"):

    true_n = len(df.index)

    pointer = ctypes.POINTER(ctypes.c_double)

    save_time = ctypes.c_double(0)
    total_time = ctypes.c_double(0)
    copy_time = ctypes.c_double(0)
    
    saveTimePtr = pointer(save_time)
    totalTimePtr = pointer(total_time)
    copyTimePtr = pointer(copy_time)

    if precision == "f4":

        n_particles,pos,vel,mass = pad(df,np.float32)

        posPtr = pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        velPtr = vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        massPtr = mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        cuda_eval_f4.cuda_evaluate(posPtr,velPtr,massPtr,ctypes.c_int(n_particles),ctypes.c_int(steps),ctypes.c_float(G),ctypes.c_float(eps),ctypes.c_float(dt),ctypes.c_int(n_params), ctypes.c_int(solver), ctypes.c_int(v), saveTimePtr, totalTimePtr, copyTimePtr)

        raw_data = np.fromfile(outname,dtype=np.float32,sep="")
    
    elif precision == "f8":
        n_particles,pos,vel,mass = pad(df,np.float64)

        posPtr = pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        velPtr = vel.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        massPtr = mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        cuda_eval_f8.cuda_evaluate(posPtr,velPtr,massPtr,ctypes.c_int(n_particles),ctypes.c_int(steps),ctypes.c_double(G),ctypes.c_double(eps),ctypes.c_double(dt),ctypes.c_int(n_params), ctypes.c_int(solver), ctypes.c_int(v), saveTimePtr, totalTimePtr, copyTimePtr)

        raw_data = np.fromfile(outname,dtype=np.float64,sep="")

    step_labels = np.repeat(np.arange(steps+1),n_particles)
    ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

    step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)
    data = pd.DataFrame(raw_data.reshape((steps+1)*n_particles,n_params),columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

    out_df = pd.concat([step_labels,ids,data],axis=1)

    stats = {}
    stats["total_save_time"] = save_time.value
    stats["total_copy_time"] = copy_time.value
    stats["total_eval_time"] = total_time.value

    os.remove('out.dat')

    return out_df[(out_df["id"] < true_n)].reset_index(),stats