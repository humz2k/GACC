import GACC
import numpy as np

n = 10000

cudaf8 = []
cudaf4 = []
ompf4 = []
ompf8 = []


df = GACC.util.Distributions.Plummer(n,M=1)

eps = 0.01
nsteps = 0

outdf3,stats4 = GACC.cuda.evaluate(df,solver=0,steps=nsteps,eps=eps)

print("f40",stats4['total_eval_time'])

outdf,stats = GACC.cuda.evaluate(df,solver=1,steps=nsteps,eps=eps)

print("f41",stats['total_eval_time'])

outdf2,stats2 = GACC.cuda.cheap_evaluate(df,solver=0,steps=nsteps,eps=eps)

print("f20",stats2['total_eval_time'])

outdf3,stats3 = GACC.cuda.cheap_evaluate(df,solver=1,steps=nsteps,eps=eps)

print("f21",stats3['total_eval_time'])
