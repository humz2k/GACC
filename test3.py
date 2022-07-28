import GACC
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os

nsteps = 1000
n = 1000
Rs = 1
p0 = 1
c = 1
dt = 1e-3
solver = 0
eps = 0
G = 1

df = GACC.util.Distributions.NFW(n,Rs,p0,c)

simDir = "/home/hqureshi/GACC/Sims/"

existing_inits = [i.split(".")[0].split("_gen")[0] for i in os.listdir(simDir) if i.endswith(".initconditions")]

name = "NFW_n" + str(n) + "_Rs" + str(Rs) + "_p0" + str(p0) + "_c" + str(c)

count = 0
if name in existing_inits:
    count = existing_inits.count(name)

name = name + "_gen" + str(count)

print(name)

df.to_parquet(simDir + name + ".initconditions")

out,stats = GACC.sim.evaluate(df,steps=nsteps,dt=dt,eps=0,solver=0)
print(stats)

out.to_parquet(simDir + name + "_dt" + str(dt).replace(".",",") + "_solver" + str(solver) + "_eps" + str(eps).replace(".",",") + "_G" + str(G).replace(".",",") + ".simulation")

