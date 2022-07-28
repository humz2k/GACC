import GACC
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os

nsteps = 1000
n = 1000
a = 1
M = 1
dt = 1e-3
solver = 0
eps = 0
G = 1

df = GACC.util.Distributions.Plummer(n,a=a,M=M)

simDir = "/home/hqureshi/GACC/Sims/"

existing_inits = [i.split(".")[0].split("_gen")[0] for i in os.listdir(simDir) if i.endswith(".initconditions")]

name = "PLUMMER_n" + str(n) + "_a" + str(a) + "_M" + str(M)

count = 0
if name in existing_inits:
    count = existing_inits.count(name)

name = name + "_gen" + str(count)

print(name)

df.to_parquet(simDir + name + ".initconditions")

out,stats = GACC.sim.evaluate(df,steps=nsteps,dt=dt,eps=0,solver=0)
print(stats)

out.to_parquet(simDir + name + "_dt" + str(dt).replace(".",",") + "_solver" + str(solver) + "_eps" + str(eps).replace(".",",") + "_G" + str(G).replace(".",",") + ".simulation")

