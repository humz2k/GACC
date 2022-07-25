import numpy as np
import pandas as pd

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

