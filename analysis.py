import pandas as pd
import GACC
import numpy as np
import matplotlib.pyplot as plt

RUN = "PLUMMER_n1000_a1_M1_gen0_dt0,0001_solver0_eps0_G1"
inputRUN = RUN.split("_dt")[0]

outdf = pd.read_parquet("/home/hqureshi/GACC/Sims/" + RUN + ".simulation")
inputdf = pd.read_parquet("/home/hqureshi/GACC/Sims/" + inputRUN + ".initconditions")

data = GACC.util.outdf2numpy(outdf)
mass = inputdf.loc[:,"mass"][0]

ke_energy = (1/2) * mass * (np.linalg.norm(data["vel"],axis=2)**2)

ke_total = np.sum(ke_energy,axis=1)

gpe_total = np.sum(data["gpe"].reshape(data["gpe"].shape[:2]),axis=1)/2

plt.title(RUN)
plt.plot(np.abs(ke_total + gpe_total),label="total")
#plt.plot(np.abs(ke_total),label="KE")
#plt.plot(np.abs(gpe_total)/2,label="GPE/2")
plt.ticklabel_format(useOffset=False)
plt.xlabel("Timestep")
plt.ylabel("Energy")
#plt.legend()
plt.tight_layout()
plt.savefig('/home/hqureshi/GACC/plots/ENERGYPLOT_' + RUN + '.jpg')