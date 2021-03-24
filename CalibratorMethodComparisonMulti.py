import os

multi_thread_Numpy = False

if not multi_thread_Numpy:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scipy as sp
from scipy.stats import uniform, norm, beta, mannwhitneyu
import pandas as pd
import numpy as np
import os
import isotpy.calibration as cal
from sklearn.metrics import brier_score_loss, mean_squared_error
from joblib import load
import multiprocessing as mp
import time
import sys

def par_expr_multi(expr): #for each training set size

    baseDir = "dataMulti"  #where to save results

    MULTI_METHODS = [cal.LogisticRegressionCalibrator(), cal.LogisticRegressionCalibrator(multi=True)]

    if not os.path.exists(baseDir):
        os.mkdir(baseDir)

    num_iters = 1000 
    nt = 10000 #number samples from each class for test set

    neg1Dist = expr[0][0]#distribution object
    neg1Info = expr[0][1] #string description of distribution
    neg1Std = expr[0][2] #sample standard deviation
    neg1Avg = expr[0][3] #sample mean

    neg2Dist = expr[1][0]
    neg2Info = expr[1][1]
    neg2Std = expr[1][2]
    neg2Avg = expr[1][3]

    pos1Dist = expr[2][0]
    pos1Info = expr[2][1]
    pos1Std = expr[2][2]
    pos1Avg = expr[2][3]


    pos2Dist = expr[3][0]
    pos2Info = expr[3][1]
    pos2Std = expr[3][2]
    pos2Avg = expr[3][3]

    for rho in [0, 0.5, 0.9]:

        #extract AUC and Distribution information to create description of experiment to use for filename
        mw = pos1Info[6:].split("]")[0]
        description = "(MW:"+mw+", rho"+str(rho)+")_"+neg1Info+"_"+neg2Info+"___"+pos1Info+"_"+pos2Info+".csv"

        #sample, shift and scale test set from negative and positive distributions
        x01_test = (neg1Dist.rvs(nt) - neg1Avg)/neg1Std
        x02_test = (neg2Dist.rvs(nt) - neg2Avg)/neg2Std
        x02_test = (x01_test * rho) + (x02_test * np.sqrt(1 - rho**2) ) #apply correlation
        x0_test = np.append(np.reshape(x01_test, (nt,1)), np.reshape(x02_test, (nt,1)), axis=1)

        x11_test = pos1Dist.rvs(nt)/pos1Std
        x12_test = pos2Dist.rvs(nt)/pos2Std
        x12_test = (x11_test * rho) + (x12_test * np.sqrt(1 - rho**2) ) #apply correlation
        x1_test = np.append(np.reshape(x11_test, (nt,1)), np.reshape(x12_test, (nt,1)), axis=1)

        indie = np.append(x0_test, x1_test, axis=0)
        y_labels_indie = np.append(np.zeros(nt), np.ones(nt)) #labels for independent test set

        #calculate true posteriors
        indie_n = (indie[:,0] * neg1Std) + neg1Avg
        fw01_indie = neg1Dist.pdf(indie_n) #find densities for negative calss
        indie_n = (indie[:,1] * neg2Std) + neg2Avg
        fw02_indie = neg2Dist.pdf(indie_n) #find densities for negative calss

        indie_n = (indie[:,0] * pos1Std)
        fw11_indie = pos1Dist.pdf(indie_n) #find densities for postive classs
        indie_n = (indie[:,1] * pos2Std)
        fw12_indie = pos2Dist.pdf(indie_n) #find densities for postive classs

        LR = (fw11_indie*fw12_indie)/( (fw01_indie*fw02_indie)+np.finfo(float).eps )
        pi = nt/(nt+nt) #calculate prior
        y_true_indie = 1 / ( 1 + (1/(LR+np.finfo(float).eps))*(1-pi)/pi ) #true posteriors for independent test set


        for n in [10,20,40,80,160,320,640,1280,2560,5120]: #for each training set size
            
            #directory for all experiments using current n value
            fullDir = baseDir+"/Multi"+str(n)+"n"
            if not os.path.exists(fullDir):
                os.mkdir(fullDir)

            #open file for brier records
            Brierfp = open(fullDir+"/Brier_"+description,"w+")
            #create header
            Brierfp.write("Metric,")
            for calibrator in MULTI_METHODS[:-1]:
                Brierfp.write(calibrator.toString()+",")
            Brierfp.write(MULTI_METHODS[-1].toString()+"\n")

            #open file for mse records
            MSEfp = open(fullDir+"/MSE_"+description,"w+")
            #create header
            MSEfp.write("Metric,")
            for calibrator in MULTI_METHODS[:-1]:
                MSEfp.write(calibrator.toString()+",")
            MSEfp.write(MULTI_METHODS[-1].toString()+"\n")

            #arrays to stores results of each iter for each method's evaluations
            brier_indie = np.zeros( (num_iters, len(MULTI_METHODS)) )
            brier_resub = np.zeros( (num_iters, len(MULTI_METHODS)) )
            mse_indie = np.zeros( (num_iters, len(MULTI_METHODS)) )
            mse_resub = np.zeros( (num_iters, len(MULTI_METHODS)) )

            y_labels_resub = np.append(np.zeros(n), np.ones(n)) #resub labels are the same each iteration

            for iter in range(num_iters): #for num_iters iterations

                start_time = time.time()

                #sample, scale, and shift train set from negative and positive distributions
                x01 = (neg1Dist.rvs(n) - neg1Avg)/neg1Std
                x02 = (neg2Dist.rvs(n) - neg2Avg)/neg2Std
                x11 = pos1Dist.rvs(n)/pos1Std
                x12 = pos2Dist.rvs(n)/pos2Std
                x02 = (x01 * rho) + (x02 * np.sqrt(1 - rho**2) )
                x0 = np.append(np.reshape(x01, (n,1)), np.reshape(x02, (n,1)), axis=1)
                x12 = (x11 * rho) + (x12 * np.sqrt(1 - rho**2) )
                x1 = np.append(np.reshape(x11, (n,1)), np.reshape(x12, (n,1)), axis=1)

                resub = np.append(x0, x1, axis=0)

                resub_n = (resub[:,0] * neg1Std) + neg1Avg
                fw01_resub = neg1Dist.pdf(resub_n) #find densities from negative calss
                resub_n = (resub[:,1] * neg2Std) + neg2Avg
                fw02_resub = neg2Dist.pdf(resub_n) #find densities from negative calss

                resub_n = (resub[:,0] * pos1Std)
                fw11_resub = pos1Dist.pdf(resub_n) #find densities from postive classs
                resub_n = (resub[:,1] * pos2Std)
                fw12_resub = pos2Dist.pdf(resub_n) #find densities from postive classs

                LR = (fw11_resub*fw12_resub)/( (fw01_resub*fw02_resub)+np.finfo(float).eps )
                pi = n/(n+n) #calculate prior
                y_true_resub = 1 / ( 1 + (1/(LR+np.finfo(float).eps))*(1-pi)/pi )  #true posteriors for train set
                
                for methodIndex, calibrator in enumerate(MULTI_METHODS): #for each calibration methods
                   
                    #train classifier and make predictions on test set
                    calibrator.train(x0,x1)
                    y_pred_indie = calibrator.test(indie)
                    y_pred_resub = calibrator.test(resub) 

                    #calculate brier score
                    brier_indie[iter][methodIndex] = brier_score_loss(y_labels_indie, y_pred_indie)
                    brier_resub[iter][methodIndex] = brier_score_loss(y_labels_resub, y_pred_resub)

                    mse_indie[iter][methodIndex] = mean_squared_error(y_true_indie,y_pred_indie)
                    mse_resub[iter][methodIndex] = mean_squared_error(y_true_resub,y_pred_resub)
                
                print(str(iter)+"--- %s seconds ---" % (time.time() - start_time))

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


exprs = load('multiExprs.joblib')

#parallel:
#   sys.argv[1] -> starting indicy of batch, used for array jobs

batchSize = 1 #numper of experiments per batch/job
pool = mp.Pool(processes=6)
results = pool.map(par_expr_multi, exprs[ int(sys.argv[1]):  int(sys.argv[1])+batchSize ])


#sequential

#map(par_expr_multi, exprs)



