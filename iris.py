__author__ = 'dan'

'''
Simple solution to the iris training dataset (for a training demo) using scikit learn (including example of how it can go bad)

'''

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

#######################
## MUNGE
#######################
# import and split data into X (features) and y (labels)
iris = datasets.load_iris()
X = iris.data
y = iris.target
print('X shape=' + str(X.shape))
print('y shape=' + str(y.shape))

# print the data grid just to make it easier to visualize
print('X data=\n' + str(X))

# create training and test data (.75/.25 split)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

#######################
## TRAIN
#######################
# create a model - in this case a simple support vector machine
clf = svm.SVC(C=1.0, class_weight=None, kernel='rbf', degree=3, gamma=0.0, random_state=None, tol=0.0001)

# train the classifier
clf.fit(X_train, y_train)

#######################
## PREDICT AND CROSS VALIDATE
#######################
# output stats on how well the training data fits the model
print('**********************************')
pred = clf.predict(X_train)
print ('TRAIN conf matrix:\n' + str(metrics.confusion_matrix(y_train, pred)))
print ('TRAIN class report:\n' + str(metrics.classification_report(y_train, pred)))

# cross validate on test set
print('**********************************')
pred = clf.predict(X_test)
print ('TEST conf matrix:\n' + str(metrics.confusion_matrix(y_test, pred)))
print ('TEST class report:\n' + str(metrics.classification_report(y_test, pred)))



# adjust some params of the model and repeat - params are worse, so results will be sub-par
clf = svm.SVC(C=.1, class_weight=None, kernel='poly', degree=6, gamma=0.013, random_state=None, tol=0.01)

# train the classifier
clf.fit(X_train, y_train)

# cross validate on test set
print('################################')
pred = clf.predict(X_test)
print ('TEST conf matrix:\n' + str(metrics.confusion_matrix(y_test, pred)))
print ('TEST class report:\n' + str(metrics.classification_report(y_test, pred)))
