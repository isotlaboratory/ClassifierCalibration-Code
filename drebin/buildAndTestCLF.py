import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import isfile, join
from bisect import bisect_left, insort
import multiprocessing as mp
import math
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_predict
from scipy.sparse import csr_matrix, save_npz, load_npz
import time
from sklearn.ensemble import RandomForestClassifier



datadir = "feature_vectors/"

rawSamples = [ join(datadir, f) for f in listdir(datadir) if isfile(join(datadir, f)) ]

fp = open("allFeatures.txt", "r")
allFeatures = fp.read().split("\n")
nFeats = len(allFeatures)
fp.close()

fp = open("Y.txt","r")
Y = fp.read().split("\n")
fp.close()

print("loading sparse feature array")
start_time = time.time()
X_sparse = load_npz('X_sparse.npz')
print("--- Sparse array loaded in %s seconds ---" % (time.time() - start_time))

#clf = svm.LinearSVC(verbose=True, class_weight="balanced")
clf = RandomForestClassifier(verbose=10, class_weight="balanced")
print("Starting CV")
y_pred = cross_val_predict(clf, X_sparse, Y, cv=10, method='predict_proba', verbose=10)

#np.save("y_predSVM.npy", y_pred)
#y_pred = np.load("y_predSVM.npy")[:,1]
np.save("y_predRF.npy", y_pred)
y_pred = np.load("y_predRF.npy")[:,1]

y_pred_neg = []
y_pred_pos = []
FN = 0
FP = 0
TP = 0
TN = 0

th = 0.119999
#FOR SVM, 0.4225  gives FP 0.010004 (about 1 in 100 FP) and Detection Rate of 0.955
#FOR RF, 0.119999  gives FP 0.0103 (about 1 in 100 FP) and Detection Rate of 0.971

for n in range(y_pred.shape[0]):
    if Y[n] == '1':
        y_pred_pos.append(y_pred[n])
        if y_pred[n] <= th:
            FN +=1
        else:
            TP +=1
    else:
        y_pred_neg.append(y_pred[n])
        if y_pred[n] > th:
            FP +=1
        else:
            TN +=1


print("miss classifications:", FN + FP)
print("Accuracy:", (y_pred.shape[0] - (FN+FP))/y_pred.shape[0])
print("Detection rate", (TP/len(y_pred_pos)))
print("FP Rate:", (FP/len(y_pred_neg)))
print("Pos scores:", len(y_pred_pos))
print("neg scores:", len(y_pred_neg))


#np.savetxt('DrebinSVMNegScores.csv', y_pred_neg, delimiter=',')
#np.savetxt('DrebinSVMPosScores.csv', y_pred_pos, delimiter=',')
np.savetxt('DrebinRFNegScores.csv', y_pred_neg, delimiter=',')
np.savetxt('DrebinRFPosScores.csv', y_pred_pos, delimiter=',')