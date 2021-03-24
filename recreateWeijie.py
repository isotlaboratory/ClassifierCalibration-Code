from scipy.stats import uniform, norm, beta
import pandas as pd
import numpy as np
import isotpy.calibration as cal
import os
from sklearn.metrics import brier_score_loss, mean_squared_error

saveDir = "plotData/weijie"

#Methods of calibration
METHODS = [cal.LogisticRegressionCalibratorPlatt(), cal.LogisticRegressionCalibrator(), cal.IsotonicRegressionCalibrator()]
if not os.path.exists(saveDir):
    os.mkdir(saveDir)

num_iters = 800 
ntest = 10000 #number samples from each class for test set

#beta and norm parameterization from Weijie
betaParams = [ [1,3.5], [1.1,1] ] 
normParams = [ [0, 1 ], [1.2,1] ] 

#--------------------------------------------Norm------------------------------------------------------------------------

#create files to save evaluation results
fpm = open(saveDir+"/"+"NormMSE"+".csv","w+")
fpb = open(saveDir+"/"+"NormBrier"+".csv","w+")

for n in [30,50,100,200,300]:

    fpm.write(str(n))
    fpb.write(str(n))

    x_0test = norm.rvs(normParams[0][0], normParams[0][1], ntest) #sample negative class test set
    x_1test = norm.rvs(normParams[1][0], normParams[1][1], ntest) #sample positive class test set
    x_inde = np.append(x_0test, x_1test) #independent test set

    for calibrator in METHODS:
        
        #resub scores
        brierSum_R = 0
        mseSum_R = 0

        #independent scores
        brierSum_I = 0
        mseSum_I = 0

        for _ in range(num_iters):

            #sample from distributions
            x_0 = norm.rvs(normParams[0][0], normParams[0][1], n)
            x_1 = norm.rvs(normParams[1][0], normParams[1][1], n)
            x_resub = np.append(x_0, x_1)


            #train classifier and make predictions on test set
            calibrator.train(x_0,x_1)
            y_resub = calibrator.test(x_resub)
            y_inde = calibrator.test(x_inde) 


            #calculate brier score
            y_true = np.append(np.zeros(n), np.ones(n))
            brierSum_R += brier_score_loss(y_true, y_resub)

            y_true = np.append(np.zeros(ntest), np.ones(ntest))
            brierSum_I += brier_score_loss(y_true, y_inde)
            

            #calculate mean_squared_error
            fw0 = norm.pdf(x_resub,  normParams[0][0], normParams[0][1]) #find densities for negative calss
            fw1 = norm.pdf(x_resub,  normParams[1][0], normParams[1][1]) #find densities for postive classs
            LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
            pi = 0.5 #calculate prior
            y_true = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #calculate true posterior probability for positive class 
            mseSum_R += mean_squared_error(y_true,y_resub) 

            fw0 = norm.pdf(x_inde,  normParams[0][0], normParams[0][1]) #find densities for negative calss
            fw1 = norm.pdf(x_inde,  normParams[1][0], normParams[1][1]) #find densities for postive classs
            LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
            pi = 0.5 #calculate prior
            y_true = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #calculate true posterior probability for positive class 
            mseSum_I += mean_squared_error(y_true,y_inde) 

        #average results over all iterations
        brier_R = brierSum_R/num_iters
        mse_R = mseSum_R/num_iters
        brier_I = brierSum_I/num_iters
        mse_I = mseSum_I/num_iters

        #write results for current n
        fpm.write(","+str(mse_R)+","+str(mse_I))
        fpb.write(","+str(brier_R)+","+str(brier_I))
    
    fpm.write("\n")
    fpb.write("\n")

fpm.close()
fpb.close()


#--------------------------------------------Beta------------------------------------------------------------------------

#create files to save evaluation scores of calibrators with current distributions
fpm = open(saveDir+"/"+"BetaMSE"+".csv","w+") #create evauation file for distribution pair
fpb = open(saveDir+"/"+"BetaBrier"+".csv","w+")

for n in [30,50,100,200,300]:

    fpm.write(str(n))
    fpb.write(str(n))

    x_0test = beta.rvs(a=betaParams[0][0], b=betaParams[0][1], size=ntest) #sample negative class test set
    x_1test = beta.rvs(a=betaParams[1][0], b=betaParams[1][1], size=ntest) #sample positive class test set
    x_inde = np.append(x_0test, x_1test) #independent test set

    for calibrator in METHODS:
        
        #resub scores
        brierSum_R = 0
        mseSum_R = 0

        #independent scores
        brierSum_I = 0
        mseSum_I = 0
        
        for _ in range(num_iters):

            #sample from distributions
            x_0 = beta.rvs(a=betaParams[0][0], b=betaParams[0][1], size=n)
            x_1 = beta.rvs(a=betaParams[1][0], b=betaParams[1][1], size=n)
            x_resub = np.append(x_0, x_1) #resubstitution test set


            #train classifier and make predictions on test set
            calibrator.train(x_0,x_1)
            y_resub = calibrator.test(x_resub)
            y_inde = calibrator.test(x_inde) 

            #calculate brier score
            y_true = np.append(np.zeros(n), np.ones(n))
            brierSum_R += brier_score_loss(y_true, y_resub)

            y_true = np.append(np.zeros(ntest), np.ones(ntest))
            brierSum_I += brier_score_loss(y_true, y_inde)
            

            #calculate mean_squared_error
            fw0 = beta.pdf(x=x_resub,  a=betaParams[0][0], b=betaParams[0][1]) #find densities for negative calss
            fw1 = beta.pdf(x=x_resub,  a=betaParams[1][0], b=betaParams[1][1]) #find densities for postive classs
            LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
            pi = 0.5 #calculate prior
            y_true = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #calculate true posterior probability for positive class 
            mseSum_R += mean_squared_error(y_true,y_resub) 

            fw0 = beta.pdf(x=x_inde,  a=betaParams[0][0], b=betaParams[0][1]) #find densities for negative calss
            fw1 = beta.pdf(x=x_inde,  a=betaParams[1][0], b=betaParams[1][1]) #find densities for postive classs
            LRrecip=fw0/(fw1+np.finfo(float).eps) #find likelihood ratio reciprocal
            pi = 0.5 #calculate prior
            y_true = 1 / ( 1 + (LRrecip)*(1-pi)/pi ) #calculate true posterior probability for positive class 
            mseSum_I += mean_squared_error(y_true,y_inde) 

        #average results over all iterations
        brier_R = brierSum_R/num_iters
        mse_R = mseSum_R/num_iters
        brier_I = brierSum_I/num_iters
        mse_I = mseSum_I/num_iters

        #write results for current n
        fpm.write(","+str(mse_R)+","+str(mse_I))
        fpb.write(","+str(brier_R)+","+str(brier_I))
    
    fpm.write("\n")
    fpb.write("\n")

fpm.close()
fpb.close()