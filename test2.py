import GACC
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os

nsteps = 2000
dt = 1e-4
G = 1
solver = 0
eps = 0

simDir = "/home/hqureshi/GACC/Sims/"

name = ""

df = pd.read_parquet(simDir + name + ".initconditions")

out,stats = GACC.sim.evaluate(df,steps=nsteps,dt=dt,eps=eps,solver=solver)
print(stats)

out.to_parquet(simDir + name + "_dt" + str(dt).replace(".",",") + "_solver" + str(solver) + "_eps" + str(eps).replace(".",",") + "_G" + str(G).replace(".",",") + ".simulation")

