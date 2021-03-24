import os

multi_thread_Numpy = False

if not multi_thread_Numpy:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scipy as sp
import pandas as pd
import numpy as np
import isotpy.calibration as cal
from sklearn.metrics import brier_score_loss, mean_squared_error
from joblib import load
import multiprocessing as mp
import time

#__________FUNCTIONS_______________________________________________________________________________________________________________________


def par_expr_mono(expr): #for each training set size

    baseDir = "dataMono" #where to save results

    N_BINS = [10, 20, 30, 40, 50]#number of bins in binning method
    METHODS = [cal.IsotonicRegressionCalibrator(),cal.LogisticRegressionCalibrator(),cal.LogisticRegressionCalibrator(multi=True)]
    for i in N_BINS:
        METHODS.append(cal.BinningCalibrator(n_bins=i))
    METHODS.append(cal.LogisticRegressionCalibratorPlatt())

    if not os.path.exists(baseDir):
        os.mkdir(baseDir)

    num_iters = 1000 
    nt = 10000 #number samples from each class for test set

    negDist = expr[0][0] #distribution object
    negInfo = expr[0][1] #string description of distribution
    negStd = expr[0][2] #sample standard deviation
    negAvg = expr[0][3] #sample mean
    
    posDist = expr[1][0]
    posInfo = expr[1][1]
    posStd = expr[1][2]
    posAvg = expr[1][3]

    #extract AUC and Distribution information to create description of experiment to use for filename
    mw = posInfo[6:].split("]")[0]
    description = "(MW:"+mw+")_"+negInfo+"_"+posInfo+".csv"


    #sample, shift and scale test set from negative and positive distribution
    x0_test = (negDist.rvs(nt) - negAvg)/negStd
    x1_test = posDist.rvs(nt)/posStd
    indie = np.append(x0_test, x1_test) #independent test set

    y_labels_indie = np.append(np.zeros(nt), np.ones(nt)) #labels for independent test set

    #calculate true posteriors for test set
    indie_n = (indie * negStd) + negAvg #scale and shift
    fw0 = negDist.pdf(indie_n) #find densities from negative distribution
    indie_n = (indie * posStd) #scale
    fw1 = posDist.pdf(indie_n) #find densities from postive distribution
    LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
    pi = nt/(nt+nt) #calculate prior
    y_true_indie = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #true posteriors for independent test set

    for n in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]: #for each training set size

        #directory for all experiments using current n value
        fullDir = baseDir+"/"+str(n)+"n"
        if not os.path.exists(fullDir):
            os.mkdir(fullDir)

        #open file for brier records
        Brierfp = open(fullDir+"/Brier_"+description,"w+")
        #create header
        Brierfp.write("Metric,")
        for calibrator in METHODS[:-1]:
            Brierfp.write(calibrator.toString()+",")
        Brierfp.write(METHODS[-1].toString()+"\n")

        #open file for mse records
        MSEfp = open(fullDir+"/MSE_"+description,"w+")
        #create header
        MSEfp.write("Metric,")
        for calibrator in METHODS[:-1]:
            MSEfp.write(calibrator.toString()+",")
        MSEfp.write(METHODS[-1].toString()+"\n")

        #arrays to stores results of each iter for each method's evaluations
        brier_indie = np.zeros( (num_iters, len(METHODS)) )
        brier_resub = np.zeros( (num_iters, len(METHODS)) )
        mse_indie = np.zeros( (num_iters, len(METHODS)) )
        mse_resub = np.zeros( (num_iters, len(METHODS)) )

        y_labels_resub = np.append(np.zeros(n), np.ones(n))#resub labels are the same each iteration

        for iter in range(num_iters): #for num_iters iterations

            start_time = time.time()

            #sample, scale, and shift train set from both distributions
            x0 = (negDist.rvs(n) - negAvg)/negStd
            x1 = posDist.rvs(n)/posStd

            resub = np.append(x0, x1)

            resub_n = (resub * negStd) + negAvg
            fw0 = negDist.pdf(resub_n) #find densities from negative calss
            resub_n = (resub * posStd)
            fw1 = posDist.pdf(resub_n) #find densities from postive classs
            LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
            pi = n/(n+n) #calculate prior
            y_true_resub = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #true posteriors for train set

            for methodIndex, calibrator in enumerate(METHODS): #for each calibration methods

                #train classifier and make predictions on train and test set
                calibrator.train(x0,x1)
                y_pred_indie = calibrator.test(indie)
                y_pred_resub = calibrator.test(resub) 

                #calculate brier score
                brier_indie[iter][methodIndex] = brier_score_loss(y_labels_indie, y_pred_indie)
                brier_resub[iter][methodIndex] = brier_score_loss(y_labels_resub, y_pred_resub)
                
                #calculate mean_squared_error
                mse_indie[iter][methodIndex] = mean_squared_error(y_true_indie,y_pred_indie)
                mse_resub[iter][methodIndex] = mean_squared_error(y_true_resub,y_pred_resub)

            print("--- %s seconds ---" % (time.time() - start_time))

        #average scores across all iterations
        brier_indie_mean = np.mean(brier_indie, axis = 0)
        brier_resub_mean = np.mean(brier_resub, axis = 0)
        mse_indie_mean = np.mean(mse_indie, axis = 0)
        mse_resub_mean = np.mean(mse_resub, axis = 0)

        #get standard deviations across all iterations
        brier_indie_std = np.std(brier_indie, axis = 0)
        brier_resub_std = np.std(brier_resub, axis = 0)
        mse_indie_std = np.std(mse_indie, axis = 0)
        mse_resub_std = np.std(mse_resub, axis = 0)

        #write scores to files and close
        MSEfp.write("mse_resub_avg,")
        for i in mse_resub_mean[:-1]:
            MSEfp.write(str(i)+",")
        MSEfp.write(str(mse_resub_mean[-1])+"\n")

        MSEfp.write("mse_indie_avg,")
        for i in mse_indie_mean[:-1]:
            MSEfp.write(str(i)+",")
        MSEfp.write(str(mse_indie_mean[-1])+"\n")

        MSEfp.write("mse_resub_std,")
        for i in mse_resub_std[:-1]:
            MSEfp.write(str(i)+",")
        MSEfp.write(str(mse_resub_std[-1])+"\n")

        MSEfp.write("mse_indie_std,")
        for i in mse_indie_std[:-1]:
            MSEfp.write(str(i)+",")
        MSEfp.write(str(mse_indie_std[-1])+"\n")
        MSEfp.close()

        Brierfp.write("brier_resub_avg,")
        for i in brier_resub_mean[:-1]:
            Brierfp.write(str(i)+",")
        Brierfp.write(str(brier_resub_mean[-1])+"\n")

        Brierfp.write("brier_indie_avg,")
        for i in brier_indie_mean[:-1]:
            Brierfp.write(str(i)+",")
        Brierfp.write(str(brier_indie_mean[-1])+"\n")

        Brierfp.write("brier_resub_std,")
        for i in brier_resub_std[:-1]:
            Brierfp.write(str(i)+",")
        Brierfp.write(str(brier_resub_std[-1])+"\n")

        Brierfp.write("brier_indie_std,")
        for i in brier_indie_std[:-1]:
            Brierfp.write(str(i)+",")
        Brierfp.write(str(brier_indie_std[-1])+"\n")
        Brierfp.close()
#__________________________________________________________________________________________________________________________________________#
#___________________________________________________________MAIN___________________________________________________________________________#
#                                                                                                                                          #

#parallel:
pool = mp.Pool(processes=6)
exprs = load('monoExprs.joblib')
results = pool.map(par_expr_mono, exprs)

#sequential

#map(par_expr_mono, exprs)
