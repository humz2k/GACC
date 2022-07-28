import GACC
import numpy as np

ns = np.arange(start = 1, stop = 10) ** 2 * 128

cudaf8 = []
cudaf4 = []
ompf4 = []
ompf8 = []

for n in ns:

    df = GACC.util.Distributions.Plummer(n)

    outdf,stats = GACC.cuda.evaluate(df,precision="f8")
    cudaf8.append(stats["total_eval_time"])

    outdf,stats = GACC.cuda.evaluate(df,precision="f4")
    cudaf4.append(stats["total_eval_time"])

    outdf,stats = GACC.omp.evaluate(df,precision="f8")
    ompf8.append(stats["total_eval_time"])

    outdf,stats = GACC.omp.evaluate(df,precision="f4")
    ompf4.append(stats["total_eval_time"])