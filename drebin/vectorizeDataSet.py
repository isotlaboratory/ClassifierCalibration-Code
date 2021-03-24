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

def maliciousStats(labels):
    print("number of malicious samples:", labels.shape[0])
    famList = list(labels[:,1])
    famfreq = {f:famList.count(f) for f in famList}
    for x, y in famfreq.items():
        print(x, y)
    print("No. Fams:", len(famfreq))

def getFeatures(range):

    features = [] 
    for samp in rawSamples[range[0]:range[1]]:
        fp = open(samp, "r")
        data = fp.read().split("\n")
        fp.close()

        for elem in data:
            if bisect_left(features, elem) < len(features):
                if features[bisect_left(features, elem)] != elem:
                    insort(features, elem)
            else:
                features.append(elem)

    return features

def getVectors(range):

    batch = range[1] - range[0]
    vectors = np.memmap('vecBuff'+range[2]+'.mmap', dtype=np.int, mode='w+', shape=(batch, len(allFeatures)) )
    #vectors = np.zeros( (len(rawSamples), len(allFeatures)) )
    labels = np.zeros( (batch, 1) ) 
    IDs = []
    for n, samp in enumerate(rawSamples[range[0]:range[1]]):
        fp = open(samp, "r")
        data = fp.read().split("\n")
        fp.close()
        for elem in data:
            vectors[n,bisect_left(allFeatures, elem)] = 1
        
        ID = samp.split("/")[1]
        if bisect_left(malIDs, ID) < len(malIDs):
            if malIDs[bisect_left(malIDs, ID)] == ID:
                labels[n] = 1
        IDs.append(ID)
        print(n)

    del vectors

    return (np.array(IDs), labels)


datadir = "feature_vectors/"

rawSamples = [ join(datadir, f) for f in listdir(datadir) if isfile(join(datadir, f)) ]

ncpus=6


step = math.ceil(len(rawSamples)/ncpus)
jobRangesandID = []
for i in range(ncpus):
    if i ==  ncpus - 1:
        jobRangesandID.append( ( i*step, len(rawSamples), str(i)) )
    else:
        jobRangesandID.append( ( i*step, (i+1)*step, str(i)) )


pool = mp.Pool(processes=ncpus)
results = pool.map(getFeatures,jobRangesandID)
pool.close()
pool.join()

for i in range(1, len(results)):
    for elem in results[i]:
        if bisect_left(results[0], elem) < len(results[0]):
                if results[0][bisect_left(results[0], elem)] != elem:
                    insort(results[0], elem)
        else:  
                results[0].append(elem)

allFeatures = results[0]
nFeats = len(allFeatures)

fp = open("allFeatures.txt", "w")
for elem in allFeatures[:-1]:
    fp.write(elem+"\n")
fp.write(allFeatures[-1])
fp.close()


malIDs = np.sort(pd.read_csv("sha256_family.csv").to_numpy()[:,0])

result = getVectors((0,len(rawSamples),'All'))
IDs = result[0]
Y = result[1]

print(Y.shape)
print(IDs.shape)

fp = open("Y.txt","w")
for i in Y:
    fp.write(str(i)+"\n")
fp.close()

fp = open("IDs.txt","w")
for i in IDs:
    fp.write(str(i)+"\n")
fp.close()

print("Opening memmap...")
X = np.memmap('vecBuffAll.mmap', dtype=np.int, mode='r', shape=(129013, 545334)) #nSamps, nFeats
print("Sparsifying memmap...")
X_sparse = csr_matrix(X)
print("Relinquishing memmap")
del X
print("Saving sparse feature array...")
save_npz('X_sparse.npz', X_sparse)