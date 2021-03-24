import os

multi_thread_Numpy = False

if not multi_thread_Numpy:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scipy as sp
from scipy.stats import uniform, norm, beta
import pandas as pd
import numpy as np
import os
import isotpy.calibration as cal
from sklearn.metrics import brier_score_loss, mean_squared_error, auc, roc_curve
from joblib import dump
import multiprocessing as mp
import time
import sys
import copy


def create_distributions(mannWhitneys, lambdaParams=[], normParams=[], betaParams=[], normalize=False):

    n = 10000

    #combine distribution information into two arrays
    negParams = []
    posParams = []
    if lambdaParams:
        lambdaNegParams = np.array(lambdaParams)[:,0]
        lambdaPosParams = np.array(lambdaParams)[:,1]
        for i in lambdaNegParams:
            negParams.append( (i, "lambda") )
        for i in lambdaPosParams:
            posParams.append( (i, "lambda") )
    
    if normParams:
        normNegParams = np.array(normParams)[:,0]
        normPosParams = np.array(normParams)[:,1]
        for i in normNegParams:
            negParams.append( (i, "norm") )
        for i in normPosParams:
            posParams.append( (i, "norm") )
    
    if betaParams:
        betaPosParams = np.array(betaParams)[:,1]
        betaNegParams = np.array(betaParams)[:,0]
        for i in betaNegParams:
            negParams.append( (i, "beta") )
        for i in betaPosParams:
            posParams.append( (i, "beta") )

    #array of negative distributions
    nds = [ [-1] ] * (len(lambdaParams)+len(normParams)+len(betaParams))
    
    #array of positive distributions
    pdsrow = [ [-1] * len(mannWhitneys) ] 
    for i in range((len(lambdaParams)+len(normParams)+len(betaParams)) - 1):
        pdsrow.append([-1] * len(mannWhitneys))
    pds = [pdsrow]
    for i in range((len(lambdaParams)+len(normParams)+len(betaParams)) - 1):
        pds.append(copy.deepcopy(pdsrow))


    for i, negPars in enumerate(negParams):           
        #create negative class distribution
        if negPars[1] == "lambda":
            Ngld = cal.GeneralizedLambdaDistribution(negPars[0][0], negPars[0][1], negPars[0][2], negPars[0][3])
            
            x_0 = Ngld.rvs(n)  #sample negative class train set for AUC
            avg0 = np.mean(x_0)
            std0 = np.std(x_0)
            if normalize:
                x_0 = (x_0 - avg0)/std0

            #add negative distribution to list of distributions
            nds[i] = ( Ngld,  "lamb_"+str(negPars[0][0])+"_"+str(negPars[0][1])+"_"+str(negPars[0][2])+"_"+str(negPars[0][3]), std0, avg0 )

        if negPars[1] == "norm":
            x_0 = norm.rvs(negPars[0][0], negPars[0][1], n) #sample negative class train set for AUC
            avg0 = np.mean(x_0)
            std0 = np.std(x_0)
            if normalize:
                x_0 = (x_0 - avg0)/std0

            #add negative distribution to list of distributions
            nds[i] = ( norm(negPars[0][0],negPars[0][1]),  "norm_"+str(negPars[0][0])+"_"+str(negPars[0][1]), std0, avg0 )

        if negPars[1] == "beta":
            x_0 = beta.rvs(negPars[0][0], negPars[0][1], size=n) #sample negative class train set for AUC
            avg0 = np.mean(x_0)
            std0 = np.std(x_0)
            if normalize:
                x_0 = (x_0 - avg0)/std0

            #add negative distribution to list of distributions
            nds[i] = ( beta(negPars[0][0], negPars[0][1]),  "beta_"+str(negPars[0][0])+"_"+str(negPars[0][1]), std0, avg0 )


        for j, mw in enumerate(mannWhitneys): #for each target AUC, determine the location parameter of the postive class distributon
            
            for k, posPars in enumerate(posParams):
                if posPars[1] == "lambda":
                    Pgld = cal.GeneralizedLambdaDistribution(posPars[0][0], posPars[0][1], posPars[0][2], posPars[0][3]) #set positive distribution's location param equal to negative distribution's
                    mw_hat, std1, avg1 = Pgld.setMeanFromMannWhitneyGLD(mw, x_0, normalize=normalize) #set the mean of the postive distribution
                    
                    pds[i][k][j] = ( Pgld,  "lamb_["+str(mw)+"]_"+str(Pgld.lam1)+"_"+str(Pgld.lam2)+"_"+str(Pgld.lam3)+"_"+str(Pgld.lam4), std1, avg1 ) #add positive distribution to list of distributions
                    
                if posPars[1] == "norm":
                    posPars[0][0], mw_hat, std1, avg1 = cal.setMeanFromMannWhitney(norm, x_0, mw, 0, posPars[0][1], normalize=normalize) #set the mean of the postive distribution
                    
                    #add positive distribution to list of distributions
                    pds[i][k][j] = ( norm(posPars[0][0],posPars[0][1]),  "norm_["+str(mw)+"]_"+str(posPars[0][0])+"_"+str(posPars[0][1]), std1, avg1 )

                if posPars[1] == "beta":
                    posPars[0][0], mw_hat, std1, avg1 = cal.setMeanFromMannWhitney(beta, x_0, mw, 0.000011, posPars[0][1], normalize=normalize) #set the location param of the postive distribution
                    
                    #add distributions to list of distributions 
                    pds[i][k][j] = ( beta(posPars[0][0], posPars[0][1]),  "beta_["+str(mw)+"]_"+str(posPars[0][0])+"_"+str(posPars[0][1]), std1, avg1 )


    monoEXPRS = []

    for n, i in enumerate(nds):

        negObj = i[0]

        for j in pds[n]:

            for m in [0,1,2]:

                posObj = j[m][0]

                #-------------------------------------------------validate AUC's------------------------------------------------------------
                x_0 = negObj.rvs(10000)  #sample negative class train set for mannywhie
                x_0 = (x_0 - i[3])/i[2]   #shift and scale
                x_1 = posObj.rvs(10000)  #sample negative class train set for mannywhie
                x_1 = (x_1)/j[m][2]          #scale
                mwhat1 = cal.mannWhitney(x_0, x_1)
                pred = np.append(x_0, x_1, axis=0)
                fpr, tpr, thresholds = roc_curve(np.append(np.zeros(1000),np.ones(1000)), pred, pos_label=1)
                mwScikit1 = auc(fpr, tpr)

                print("\tMW:",mwhat1,"\tMW-SK:",mwScikit1)
                #------------------------------------------------------------------------------------------------------------------------------

                monoEXPRS.append([i, j[m]])


    dump(monoEXPRS, 'monoExprs.joblib')

    multiEXPRs = []

    for n1, i1 in enumerate(nds):

        negObj1 = i1[0]

        for n2, i2 in enumerate(nds):

            negObj2 = i2[0]

            for j1 in pds[n1]:

                for j2 in pds[n2]:

                    for m in [0,1,2]:

                        posObj1 = j1[m][0]
                        posObj2 = j2[m][0]

                        #-------------------------------------------------validate AUC's------------------------------------------------------------
                        x_0 = negObj1.rvs(10000)  #sample negative class train set for mannywhie
                        x_0 = (x_0 - i1[3])/i1[2]   #shift and scale
                        x_1 = posObj1.rvs(10000)  #sample negative class train set for mannywhie
                        x_1 = (x_1)/j1[m][2]          #scale
                        mwhat1 = cal.mannWhitney(x_0, x_1)
                        pred = np.append(x_0, x_1, axis=0)
                        fpr, tpr, thresholds = roc_curve(np.append(np.zeros(1000),np.ones(1000)), pred, pos_label=1)
                        mwScikit1 = auc(fpr, tpr)

                        x_0 = negObj2.rvs(10000)  #sample negative class train set for mannywhie
                        x_0 = (x_0 - i2[3])/i2[2]   #shift and scale
                        x_1 = posObj2.rvs(10000)  #sample negative class train set for mannywhie
                        x_1 = (x_1)/j2[m][2]          #scale
                        mwhat2 = cal.mannWhitney(x_0, x_1)
                        pred = np.append(x_0, x_1, axis=0)
                        fpr, tpr, thresholds = roc_curve(np.append(np.zeros(1000),np.ones(1000)), pred, pos_label=1)
                        mwScikit2 = auc(fpr, tpr)

                        print("\tMW1:",mwhat1,"\tMW-SK1:",mwScikit1,"\tMW2:",mwhat2,"\tMW-SK2:",mwScikit2)
                        #-------------------------------------------------------------------------------------------------------------------------------

                        multiEXPRs.append([i1, i2, j1[m], j2[m]])

    dump(multiEXPRs, 'multiExprs.joblib')
    
    return


#---------MAIN---------------------------------------------------------------------------------------------------------------------------------------------------

print("\n")

baseDir = "dataFinalmulti"

NORMALIZE = True

mannWhitneys = [0.6, 0.75, 0.9]

lambdaParams = [ [[0,0.014,0.009695,0.0285], [0.46875,0.014,0.009695,0.0285]],          #right
                 [[0,0.014,0.0285,0.009695], [1.3575,0.014,0.0285,0.009695]],           #left
                 [[0,-0.1125,-0.1359,-0.1359], [1.6611,-0.1125,-0.1359,-0.1359]] ]      #symmetric heavy tails

normParams = [  [ [0,1], [1.2,1] ] ]

create_distributions(mannWhitneys,lambdaParams=lambdaParams,normParams=normParams, normalize=NORMALIZE)

