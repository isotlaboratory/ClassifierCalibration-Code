from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X1 = pd.read_csv("spamVectors/testinstancesham.csv").to_numpy() #ham 28844
X2 = pd.read_csv("spamVectors/testinstancespam.csv").to_numpy() #spam 722451


print("dataloaded")

#remove lebel column 
labelIndex = 0
for i in range(X1.shape[1]):
    if X1[0,i] != 1 and X1[0,i] != 0:
        labelIndex = i
        break
X1 = np.delete(X1, labelIndex, 1)

#remove lebel column 
labelIndex = 0
for i in range(X2.shape[1]):
    if X2[0,i] != 1 and X2[0,i] != 0:
        labelIndex = i
        break
X2 = np.delete(X2, labelIndex, 1)


X = np.append(X1, X2, axis=0) #append ham and spam
Y = np.append( np.zeros( (X1.shape[0],1) ), np.ones( (X2.shape[0],1) ) ) #labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50, random_state=10)

np.savetxt('X_train.csv', X_train, delimiter=',')
p.savetxt('X_test.csv', X_test, delimiter=',')
np.savetxt('Y_train.csv', Y_train, delimiter=',')
np.savetxt('Y_test.csv', Y_test, delimiter=',')