import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import datasets
from sklearn import svm

iris=datasets.load_iris()
#print(iris.target)
X_train,X_test,Y_train,Y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=0)
clf=svm.SVC(kernel='linear',C=1).fit(X_train,Y_train)
score=clf.score(X_test,Y_test)
#print(score)
kfld=cross_val_score(clf,iris.data,iris.target,cv=6)
print(kfld)
print(kfld.mean())
clf1=svm.SVC(kernel='poly',C=1,degree=3).fit(X_train,Y_train)
score1=clf.score(X_test,Y_test)
#print(score1)
kfld1=cross_val_score(clf1,iris.data,iris.target,cv=6)
print(kfld1,kfld1.mean())