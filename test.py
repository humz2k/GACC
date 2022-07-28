import sim
import util
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

testdf = util.Distributions.Uniform(32)
sim.evaluate(testdf,steps=0,dt=1/64,solver=1)

ns = np.arange(start=1,stop=20,step=10)**2 * 128
nsteps = 0

ys = []
xs = []
save_ys = []

for n in ns:
    
    for i in range(2):
        print(n)
        df = util.Distributions.Plummer(n)
        first = time.perf_counter()
        out,stats = sim.evaluate(df,steps=nsteps,dt=1/64,solver=0)
        second = time.perf_counter()
        print(stats)
        #save_ys.append(save_time)
        #xs.append(n)
        #ys.append(second-first)
        #print(save_time)

'''
plt.scatter(xs,ys,label="execution")
plt.scatter(xs,save_ys,label="save")
plt.xlabel('n')
plt.ylabel('execution time')
plt.legend()
plt.tight_layout()
plt.savefig('time.jpg')
'''

#first1 = time.perf_counter()
#out1 = sim.evaluate(df,steps=nsteps,dt=1/64,solver=0)
#second1 = time.perf_counter()

#error = np.abs(out.loc[:,["ax","ay","az","gpe"]].to_numpy() - out1.loc[:,["ax","ay","az","gpe"]].to_numpy())
#print(error[-10:])

#print(second-first,second1-first1)