import GACC
import numpy as np

n = 32*100

cudaf8 = []
cudaf4 = []
ompf4 = []
ompf8 = []


df = GACC.util.Distributions.Plummer(n,M=50)

print(df)

'''
outdf,stats = GACC.cuda.evaluate(df,precision="f4")
ompf4.append(stats["total_eval_time"])

outdf1,stats1 = GACC.cuda.half_evaluate(df)
ompf4.append(stats["total_eval_time"])
'''

nsteps = 10

outdf2,stats2 = GACC.cuda.cheap_evaluate(df,solver=0,steps=nsteps,eps=1)

outdf3,stats3 = GACC.cuda.cheap_evaluate(df,solver=1,steps=nsteps,eps=1)

outdf,stats = GACC.cuda.evaluate(df,solver=1,steps=nsteps,eps=1)

#print(outdf3[outdf3["step"]==1])

#print(outdf2[outdf2["step"]==1])

print(outdf2[outdf2["id"] == 0])
print(outdf2[outdf2["id"] == 1])
print(outdf[outdf["id"] == 0])