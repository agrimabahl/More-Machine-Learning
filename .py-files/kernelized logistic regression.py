import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle


def logistic(x):
    return 1/(1 + np.exp(-x))

def risk(a,kern,X,Y):
    risk_f=0
    for i in range(0,len(X)):
        #wx=w.transpose().dot(X[i])
        Ka=a.dot(kern[i])
        sig=logistic(Ka)
        if sig==1:
            sig=0.99999995
        if sig==0:
            sig=0.00000005
        risk_f+=-(Y[i]*np.log(sig)+(1-Y[i])*np.log(1-sig))
    return risk_f

def kernel(X,Y):
    a=np.zeros(len(X),dtype='float')
    a[0]=1
    #a=np.random.rand
    rho=0.01
    kern=kern_matrix(X,X,rho)
    #maxiter=1000
    eps=0.000001
    risk_fn=np.zeros(len(X)/50+1,dtype='float')
    for i in range(len(X)):
        Ka=a.dot(kern[i])
        sig=logistic(Ka)
        if sig==1:
            sig=0.99999995
        if sig==0:
            sig=0.00000005
        if i%50 ==0:
            j=int(i/50)
            #risk_fn[j]=risk(w,X,Y)
            risk_fn[j]=risk(a,kern,X,Y)
        a=a+eps*(Y[i]-logistic(Ka))
        print("at iter",i)
        print("Risk fn value=",risk_fn[j])
        print("dual weights are",a)
    return risk_fn,a    

def k(x,z,rho):
    kern=x.transpose().dot(z)+rho
    return kern

def kern_matrix(X,Z,rho):
    kern_mat=np.zeros(shape=(len(X),len(Z)))
    for i in range(len(X)):
        for j in range(len(Z)):
            kern_mat[i][j]=k(X[i],Z[j],rho)
    return kern_mat                            

def predict(a,Xtrain, Xvalid):
    pred_values=np.zeros(len(Xvalid))
    kern=kern_matrix(Xvalid,Xtrain,1)
    for i in range(len(Xvalid)):
        Ka=a.dot(kern[i])
        pred=logistic(Ka)
        if pred<=0.5:
            pred_values[i]=0
        else:
            pred_values[i]=1
    return pred_values        

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices


def main():
	tr = scipy.io.loadmat('spam_data.mat')
	Xtrain=tr["training_data"]
	Ytrain=tr["training_labels"][0]
	Xtest=tr["test_data"]

	#Pre-Processing the training matrix by normalizing (mean=1,sd=0)
	Xtrain=np.asfarray(Xtrain)
	for i in range(0,32):
    	Xtrain[:,i]=preprocessing.scale(Xtrain[:,i])

    #adding bias	
    Xtrain = np.insert(Xtrain, 32, values=1, axis=1)
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=0)

    Xtraining=Xtrain2[0:3448]
    Ytraining=Ytrain[0:3448]
    Xvalid=Xtrain2[3448:]
	Yvalid=Ytrain[3448:]

	risk_func,weights=kernel(Xtraining,Ytraining)

	#Risk function plot below
	x = [i*50 for i in range(len(risk_func))]
	plt.plot(x,risk_func)
	plt.ylabel('Risk function')
	plt.xlabel('Iterations')
	plt.show()

	#predicting on the validation set
	pred_val=predict(weights, Xtraining, Xvalid)
	benchmark(np.array(pred_val),np.array(Yvalid))



	
