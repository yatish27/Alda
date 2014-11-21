from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy
import scipy as sp
#from skll.metrics import kappa


#Evaluation
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, classification_report, f1_score, precision_score, recall_score, roc_auc_score

import sklearn.metrics


def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

dataset = genfromtxt(open('train50k_16bit_missing_vals.csv','r'), delimiter=',', dtype='f8')
#xvector
X = [x[2:] for x in dataset]
#ylabel
Y = [x[1] for x in dataset]

X = numpy.asarray(X)
Y = numpy.asarray(Y)

#Random Forest
rf = RandomForestClassifier(n_estimators=10)
#rf.fit(X, Y)


#scores = cross_val_score(rf, X, Y, scoring='accuracy')
#print(scores)


cv = cross_validation.KFold(len(X), n_folds=10)
correct_classified = []
mae = []
mse = []
ps = []
recall = []
roc_auc = []
f1 = []

for train_index, test_index in cv:
    rf.fit(X[train_index], Y[train_index])

    probas      = rf.predict_proba(X[test_index])
    probas_ceil = rf.predict(X[test_index])

    print("Correctly Classified Instances: %d" %accuracy_score(Y[test_index], probas_ceil, normalize=False))
    correct_classified.append(accuracy_score(Y[test_index], probas_ceil, normalize=False))

    print("Mean AE: %f" %mean_absolute_error(Y[test_index], probas_ceil))
    mae.append(accuracy_score(Y[test_index], probas_ceil))

    print("Mean SE: %f" %mean_squared_error(Y[test_index], probas_ceil))
    mse.append(mean_squared_error(Y[test_index], probas_ceil))

    print("Precision Score: %f" %precision_score(Y[test_index], probas_ceil))
    ps.append(precision_score(Y[test_index], probas_ceil))

    print("Recall Score: %f" %recall_score(Y[test_index], probas_ceil))
    recall.append(recall_score(Y[test_index], probas_ceil, average='weighted'))

    print("Area ROC: %f" %roc_auc_score(Y[test_index], probas_ceil))
    roc_auc.append(roc_auc_score(Y[test_index], probas_ceil))

    print("F1 Score: %f" %f1_score(Y[test_index], probas_ceil, average='weighted'))
    f1.append(f1_score(Y[test_index], probas_ceil))

    print(classification_report(Y[test_index], probas_ceil))

    cm = confusion_matrix(Y[test_index], probas_ceil)
    print(cm)
    #results.append( llfun(Y[test_index], [x[1] for x in probas]) )

#print "Results: " + str( numpy.array(results).mean() )
