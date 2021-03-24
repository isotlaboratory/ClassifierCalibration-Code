from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from joblib import dump, load
import numpy as np


X_train = np.loadtxt('X_train.csv', delimiter=',')
X_test = np.loadtxt('X_test.csv', delimiter=',')
Y_train = np.loadtxt('Y_train.csv', delimiter=',')
Y_test = np.loadtxt('Y_test.csv', delimiter=',')

print("dataloaded")

#------------------------------------------------------------------RF------------------------------------------------------------------------------


clf = RandomForestClassifier(n_estimators=200)
cv_results = cross_validate(clf, X_train, Y_train, verbose=10, cv=10, scoring='balanced_accuracy', n_jobs=6, return_estimator=True)

m, mn = 0, 0
for n, i in enumerate(cv_results['test_score']):
    if i > m:
        m = i
        mn = n
print("Build RF Done! Accuracy on left out fold:", m)

best_clf = cv_results['estimator'][mn]
dump(best_clf, "best_classifierRF.joblib")


y_pred = best_clf.predict(X_test)
print("Accuracy on test set:",accuracy_score(Y_test, y_pred))
print("Balanced accuracy on test set:",balanced_accuracy_score(Y_test, y_pred))

y_prob = best_clf.predict_proba(X_test)

fp = open("RFscores.csv", "w+")
for i in y_prob:
    fp.write(str(i)+"\n")
fp.close()

#------------------------------------------------------------------SVM------------------------------------------------------------------------------

clf = SVC(degree=2, probability=True)
cv_results = cross_validate(clf, X_train, Y_train, verbose=10, cv=10, scoring='balanced_accuracy', n_jobs=6, return_estimator=True)

m, mn = 0, 0
for n, i in enumerate(cv_results['test_score']):
    if i > m:
        m = i
        mn = n
print("Build SVM Done! Accuracy on left out fold:", m)

best_clf = cv_results['estimator'][mn]
dump(best_clf, "best_classifierSVM.joblib")


y_pred = best_clf.predict(X_test)
print("Accuracy on test set:",accuracy_score(Y_test, y_pred))
print("Balanced accuracy on test set:",balanced_accuracy_score(Y_test, y_pred))

y_prob = best_clf.decision_function(X_test)

fp = open("SVMscores.csv", "w+")
for i in y_prob:
    fp.write(str(i)+"\n")
fp.close()

