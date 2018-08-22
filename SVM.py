from sklearn import  datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def main():

    data=datasets.load_iris()
    X=data.data[:, :2]
    #print(X[:,0])
    #print (x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1)
    svc=svm.SVC(kernel='linear',C=1.0).fit(X,data.target)
    rbf=svm.SVC(kernel='rbf',gamma='auto',C=1).fit(X,data.target)
    poly_svc=svm.SVC(kernel='poly', degree=3,C=1).fit(X,data.target)
    plotPrediction(svc,X,data.target)
    plotPrediction(rbf, X, data.target)
    plotPrediction(poly_svc, X, data.target)

def plotPrediction(srp,X,y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #print(x_min,x_max,y_max,y_min)
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
    Z=srp.predict(np.c_[xx.ravel(),yy.ravel()])
    #plt.figure(figsize=(8,6))
    Z=Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)
    plt.scatter(X[:,0],X[:,1],c=y.astype(np.float))
    plt.show()
    print(srp.predict([[7.1,4.0]]))

if __name__=="__main__" :
    main()