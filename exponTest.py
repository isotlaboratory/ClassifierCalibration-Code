

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import isotpy.calibration as cal
import os
import time
from sklearn.metrics import brier_score_loss, mean_squared_error, auc, roc_curve
import isotpy.calibration as cal


exprs = load("exponExprs.joblib")

baseDir = "dataExpon"

if not os.path.exists(baseDir):
    os.mkdir(baseDir)

METHODS = [cal.IsotonicRegressionCalibrator(),cal.LogisticRegressionCalibrator(),cal.LogisticRegressionCalibrator(multi=True)]
for i in [10, 20, 30, 40, 50]:
    METHODS.append(cal.BinningCalibrator(n_bins=i))
METHODS.append(cal.LogisticRegressionCalibratorPlatt())

num_iters = 1
nt = 1000000

for expr in exprs[-2:-1]:

    negDist = expr[0] #negative distribution object
    posDist = expr[1] #positive distribution object
    description = expr[2] #string description of distribution

    #sample, shift and scale test set from negative and positive distribution
    x0_test = negDist.rvs(nt) 
    x1_test = posDist.rvs(nt)
    indie = np.append(x0_test, x1_test) #independent test set

    y_labels_indie = np.append(np.zeros(nt), np.ones(nt))  #labels for independent test set

    #calculate true posteriors for test set
    fw0 = negDist.pdf(indie) #find densities for negative calss
    fw1 = posDist.pdf(indie) #find densities for postive classs
    LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
    pi = nt/(nt+nt) #calculate prior
    y_true_indie = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #true posteriors for independent test set

    for n in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]: #for each training set size

        #create diretory for data
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

            #sample train set from both distributions
            x0 = negDist.rvs(n)
            x1 = posDist.rvs(n)

            resub = np.append(x0, x1)

            fw0 = negDist.pdf(resub) #find densities from negative calss
            fw1 = posDist.pdf(resub) #find densities from postive classs
            LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
            pi = n/(n+n) #calculate prior
            y_true_resub = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #true posteriors for train set

            for methodIndex, calibrator in enumerate(METHODS): #for each calibration methods

                #train classifier and make predictions on train and test set
                calibrator.train(x0, x1)
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