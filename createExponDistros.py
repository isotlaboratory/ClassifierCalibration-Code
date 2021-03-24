import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
import isotpy.calibration as cal


exprs = []

for mw in [0.60,0.75,0.90,0.99]: #for each AUC

    res = 10000000 #resolution for finding distribution histogram
    nbins = 100 #number of bins for histogram

    #initialize distributions
    negDist = cal.exponentialDistribution()
    posDist = cal.exponentialDistribution(flip=True)

    #set lambda param so AUC equals target mw
    mw, scale = cal.setLambdasFromMannWhitneyExpon(mw, negDist, posDist, step=0.1)

    #plot distributions
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 10000)

    ax.plot(x,negDist.pdf(x))
    ax.plot(x,posDist.pdf(x))

    sample_N = negDist.rvs(res)
    ax.hist(sample_N,  bins=nbins, alpha=0.5, density=True)

    sample_P = posDist.rvs(res)
    ax.hist(sample_P,  bins=nbins, alpha=0.5, density=True)

    plt.show()
    plt.close()

    #append experiment to list of experiments
    exprs.append([negDist, posDist, "expo_"+str(negDist.lam)])

dump(exprs, "exponExprs.joblib")