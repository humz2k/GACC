import GACC
import numpy as np

n = 32*100

cudaf8 = []
cudaf4 = []
ompf4 = []
ompf8 = []


df = GACC.util.Distributions.Plummer(n,M=1)

print(df)

'''
outdf,stats = GACC.cuda.evaluate(df,precision="f4")
ompf4.append(stats["total_eval_time"])

outdf1,stats1 = GACC.cuda.half_evaluate(df)
ompf4.append(stats["total_eval_time"])
'''
eps = 0.01
nsteps = 100

outdf2,stats2 = GACC.cuda.cheap_evaluate(df,solver=0,steps=nsteps,eps=eps)

outdf3,stats3 = GACC.cuda.cheap_evaluate(df,solver=1,steps=nsteps,eps=eps)

outdf,stats = GACC.cuda.evaluate(df,solver=1,steps=nsteps,eps=eps)

#print(outdf3[outdf3["step"]==1])

#print(outdf2[outdf2["step"]==1])
print(outdf3)
print(outdf2)
print(np.mean((np.abs((outdf2 - outdf).to_numpy())/np.abs(outdf.to_numpy()))[:,[2,3,4,5,6,7,8,9]]))
print(np.mean((np.abs((outdf3 - outdf).to_numpy())/np.abs(outdf.to_numpy()))[:,[2,3,4,5,6,7,8,9]]))