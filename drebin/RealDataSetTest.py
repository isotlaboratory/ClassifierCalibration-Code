import isotpy.calibration as cal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, auc, roc_curve
from joblib import dump, load
from sklearn.utils import class_weight


#initialize array of sample set sizes
n_test = [10,20,40,80,160,320,640,1280,2560,5120]

#initialize methods for calibration
METHODS = [cal.IsotonicRegressionCalibrator(),cal.LogisticRegressionCalibrator(),cal.LogisticRegressionCalibrator(multi=True)]
for i in [10, 20, 30, 40, 50]:
    METHODS.append(cal.BinningCalibrator(n_bins=i))
METHODS.append(cal.LogisticRegressionCalibratorPlatt())

#--------RANDOM FOREST

#load probabilities for Random Forest
x0 = np.loadtxt('DrebinRFNegScores.csv', delimiter=',')
x1 = np.loadtxt('DrebinRFPosScores.csv', delimiter=',')

#split into train distribution and test set, 
#   x0 will be sampled from for training
#   x0test is fixed test set
x0, x0test = train_test_split(x0, train_size=0.5, random_state=21)


#Calculate Mann-Whiteney for Random Forest
fpr, tpr, thresholds = roc_curve(np.append(np.zeros((x0.shape[0], 1)), np.ones((x1.shape[0], 1)) ), np.append(x0, x1), pos_label=1)
print("RF Mann-Whitney:", auc(fpr, tpr))

#initialize array of brier scores for resubstition and independent test set
brier_indie = np.ones( (len(n_test), len(METHODS)) )
brier_resub = np.ones( (len(n_test), len(METHODS)) )

for m_ind, calibrator in enumerate(METHODS): #for each calibrator method
    for n_ind, n in enumerate(n_test): #for each training set size
        
        #sum of brier scores
        brier_indie_sum = 0
        brier_resub_sum = 0
        for _ in range(1000):#for each iteration
            print(_)
            x0train, _split = train_test_split(x0, train_size=int(n)) #sample training set for negative classes
            x1train, x1test = train_test_split(x1, train_size=int(n)) #sample training set for positive classes

            #create labels
            y_labels_resub = np.append(np.zeros((int(n), 1)), np.ones((int(n),1)) , axis=0)
            y_labels_indie = np.append(np.zeros((x0test.shape[0], 1)), np.ones((x1test.shape[0],1)) , axis=0)

            #train calibrator
            calibrator.train(x0train,x1train)

            #create resubsitution set
            resub = np.append(x0train, x1train, axis=0)
            #create independent set
            indie = np.append(x0test, x1test, axis=0)

            #make predictions on independent and resubstituion sets
            y_pred_indie = calibrator.test(indie)
            y_pred_resub = calibrator.test(resub) 

            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels_indie), y_labels_indie[:,0])
            sample_weights = np.append(np.repeat(class_weights[0], x0test.shape[0]),  np.repeat(class_weights[1], x1test.shape[0]))


            #calculate brier scores
            brier_indie_sum += np.sqrt(brier_score_loss(y_labels_indie, y_pred_indie, sample_weight = sample_weights))
            brier_resub_sum += np.sqrt(brier_score_loss(y_labels_resub, y_pred_resub))

        #average out brier scores
        brier_indie[n_ind][m_ind] = brier_indie_sum/1000
        brier_resub[n_ind][m_ind] = brier_resub_sum/1000

#save brier scores
fp = open("DrebinRFIndieRoot.csv","w+")
for i in range(brier_indie.shape[0]):
    fp.write(str(n_test[i])+",")
    for j in range(brier_indie.shape[1]-1):
        fp.write(str(brier_indie[i,j])+",")
    fp.write(str(brier_indie[i,-1])+"\n")

fp = open("DrebinRFResubRoot.csv","w+")
for i in range(brier_resub.shape[0]):
    fp.write(str(n_test[i])+",")
    for j in range(brier_resub.shape[1]-1):
        fp.write(str(brier_resub[i,j])+",")
    fp.write(str(brier_resub[i,-1])+"\n")

#--------SVM

#load probabilities for SVM
x0 = np.loadtxt('DrebinSVMNegScores.csv', delimiter=',')
x1 = np.loadtxt('DrebinSVMPosScores.csv', delimiter=',')

#split into train distribution and test set, 
#   x0 will be sampled from for training
#   x0test is fixed test set
x0, x0test = train_test_split(x0, train_size=0.5, random_state=21)


#Calculate Mann-Whiteney for SVM
fpr, tpr, thresholds = roc_curve(np.append(np.zeros((x0.shape[0], 1)), np.ones((x1.shape[0], 1)) ), np.append(x0, x1), pos_label=1)
print("SVM Mann-Whitney:", auc(fpr, tpr))

#initialize array of brier scores for resubstition and independent test set
brier_indie = np.ones( (len(n_test), len(METHODS)) )
brier_resub = np.ones( (len(n_test), len(METHODS)) )

for m_ind, calibrator in enumerate(METHODS): #for each calibrator method
    for n_ind, n in enumerate(n_test): #for each training set size
        
        #sum of brier scores
        brier_indie_sum = 0
        brier_resub_sum = 0
        for _ in range(1000):#for each iteration
            print(_)
            x0train, _split = train_test_split(x0, train_size=int(n)) #sample training set for negative classes
            x1train, x1test = train_test_split(x1, train_size=int(n)) #sample training set for positive classes

            #create labels
            y_labels_resub = np.append(np.zeros((int(n), 1)), np.ones((int(n),1)) , axis=0)
            y_labels_indie = np.append(np.zeros((x0test.shape[0], 1)), np.ones((x1test.shape[0],1)) , axis=0)

            #train calibrator
            calibrator.train(x0train,x1train)

            #create resubsitution set
            resub = np.append(x0train, x1train, axis=0)
            #create independent set
            indie = np.append(x0test, x1test, axis=0)

            #make predictions on independent and resubstituion sets
            y_pred_indie = calibrator.test(indie)
            y_pred_resub = calibrator.test(resub) 

            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels_indie), y_labels_indie[:,0])
            sample_weights = np.append(np.repeat(class_weights[0], x0test.shape[0]),  np.repeat(class_weights[1], x1test.shape[0]))


            #calculate brier scores
            brier_indie_sum += np.sqrt(brier_score_loss(y_labels_indie, y_pred_indie, sample_weight = sample_weights))
            brier_resub_sum += np.sqrt(brier_score_loss(y_labels_resub, y_pred_resub))

        #average out brier scores
        brier_indie[n_ind][m_ind] = brier_indie_sum/1000
        brier_resub[n_ind][m_ind] = brier_resub_sum/1000

#save brier scores
fp = open("DrebinSVMIndieRoot.csv","w+")
for i in range(brier_indie.shape[0]):
    fp.write(str(n_test[i])+",")
    for j in range(brier_indie.shape[1]-1):
        fp.write(str(brier_indie[i,j])+",")
    fp.write(str(brier_indie[i,-1])+"\n")

fp = open("DrebinSVMResubRoot.csv","w+")
for i in range(brier_resub.shape[0]):
    fp.write(str(n_test[i])+",")
    for j in range(brier_resub.shape[1]-1):
        fp.write(str(brier_resub[i,j])+",")
    fp.write(str(brier_resub[i,-1])+"\n")

#--------MULTI

#separate train and test classes for both SVM and Random Forest
x01 = np.loadtxt('DrebinRFNegScores.csv', delimiter=',')
x11 = np.loadtxt('DrebinRFPosScores.csv', delimiter=',')

x02 = np.loadtxt('DrebinSVMNegScores.csv', delimiter=',')
x12 = np.loadtxt('DrebinSVMPosScores.csv', delimiter=',')


#split into train distribution and test set, 
#   x0 will be sampled from for training
#   x0test is fixed test set
x01, x0test1 = train_test_split(x01, train_size=0.5, random_state=21)
x02, x0test2 = train_test_split(x02, train_size=0.5, random_state=21)

#initialize methods for calibration
MULTI_METHODS = [cal.LogisticRegressionCalibrator(), cal.LogisticRegressionCalibrator(multi=True)]

#reset array of brier scores for resubstition and independent test set
brier_indie = np.zeros( (len(n_test), len(MULTI_METHODS)) )
brier_resub = np.zeros( (len(n_test), len(MULTI_METHODS)) )

for m_ind, calibrator in enumerate(MULTI_METHODS):#for each calibrator method
    for n_ind, n in enumerate(n_test):#for each training set size
        
        #sum of brier scores
        brier_indie_sum = 0
        brier_resub_sum = 0
        for _ in range(1000):#for each iteration

            x0train1, _split = train_test_split(x01, train_size=n)#sample training set for negative classes for SVM
            x1train1, x1test1 = train_test_split(x11, train_size=n)#sample training set for Positive classes for SVM

            x0train2, _split = train_test_split(x02, train_size=n)#sample training set for negative classes for RandomForest
            x1train2, x1test2 = train_test_split(x12, train_size=n)#sample training set for Positive classes for RandomForest

            #create labels
            y_labels_resub = np.append(np.zeros((n, 1)), np.ones((n,1)) , axis=0)
            y_labels_indie = np.append(np.zeros((x0test1.shape[0], 1)), np.ones((x1test1.shape[0],1)) , axis=0)


            #create resubsitution set
            x0train = np.append(np.reshape(x0train1, (x0train1.shape[0],1)), np.reshape(x0train2, (x0train2.shape[0],1)), axis=1)
            x1train = np.append(np.reshape(x1train1, (x1train1.shape[0],1)), np.reshape(x1train2, (x1train2.shape[0],1)), axis=1)

            resub = np.append(x0train, x1train, axis=0)

            #create independent set
            x0test = np.append(np.reshape(x0test1, (x0test1.shape[0],1)), np.reshape(x0test2, (x0test2.shape[0],1)), axis=1)
            x1test = np.append(np.reshape(x1test1, (x1test1.shape[0],1)), np.reshape(x1test2, (x1test2.shape[0],1)), axis=1)

            indie = np.append(x0test, x1test, axis=0)

            #make predictions on independent and resubstituion sets
            calibrator.train(x0train,x1train)
            y_pred_indie = calibrator.test(indie)
            y_pred_resub = calibrator.test(resub) 

            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels_indie), y_labels_indie[:,0])
            sample_weights = np.append(np.repeat(class_weights[0], x0test1.shape[0]),  np.repeat(class_weights[1], x1test1.shape[0]))

            #calculate brier score
            brier_indie_sum += np.sqrt(brier_score_loss(y_labels_indie, y_pred_indie, sample_weight = sample_weights))
            brier_resub_sum += np.sqrt(brier_score_loss(y_labels_resub, y_pred_resub))

            print(_)

        #average out brier scores
        brier_indie[n_ind][m_ind] = brier_indie_sum/1000
        brier_resub[n_ind][m_ind] = brier_resub_sum/1000

#save brier scores
fp = open("DrebinMultiIndieRoot.csv","w+")
for i in range(brier_indie.shape[0]):
    fp.write(str(n_test[i])+",")
    for j in range(brier_indie.shape[1]-1):
        fp.write(str(brier_indie[i,j])+",")
    fp.write(str(brier_indie[i,-1])+"\n")

fp = open("DrebinMultiResubRoot.csv","w+")
for i in range(brier_resub.shape[0]):
    fp.write(str(n_test[i])+",")
    for j in range(brier_resub.shape[1]-1):
        fp.write(str(brier_resub[i,j])+",")
    fp.write(str(brier_resub[i,-1])+"\n")
