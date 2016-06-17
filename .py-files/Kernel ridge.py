%pylab inline
import matplotlib.pyplot as plt
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
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

def kernel_ridge(X,Y):
    a=np.zeros(len(X))
    a[0]=1
    #a=np.random.rand
    rho=0.1
    lamb=0.0001
    kern=kern_matrix(X,X,rho)
    #maxiter=1000
    eps=0.000001
    risk_fn=np.zeros(len(X)/50+1,dtype='float')
    for i in range(len(X)):
        Ka=a.dot(kern[i])
        #sig=logistic(Ka)
        #if sig==1:
        #    sig=0.99999995
        #if sig==0:
        #    sig=0.00000005
        if i%50 ==0:
            j=int(i/50)
            #risk_fn[j]=risk(w,X,Y)
            risk_fn[j]=risk(a,kern,X,Y)
        a=a+eps*(Y[i]-logistic(Ka))-lamb*a
        print("at iter",i)
        print("Risk fn value=",risk_fn[j])
        print("dual weights are",a)
    return risk_fn,a 

def kern_matrix(X,Z,rho):
    kern_mat=np.zeros(shape=(len(X),len(Z)))
    for i in range(len(X)):
        for j in range(len(Z)):
            kern_mat[i][j]=k(X[i],Z[j],rho)
    return kern_mat        


def k(x,z,rho):
    kern=x.transpose().dot(z)+rho
    return kern*kern

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
	Xtrain = np.insert(Xtrain, 32, values=1, axis=1)
	Xtrain=tr["training_data"]
	Ytrain=tr["training_labels"][0]
	Xtest=tr["test_data"]

	#Pre-Pro (iii)
	for i in range(0,5172):
    	for j in range(0,32):
        	if Xtrain[i][j]>0:
            	Xtrain[i][j]=1
        	else:
            	Xtrain[i][j]=0 

	Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=0)
	Xtraining=Xtrain[0:3448]
	Ytraining=Ytrain[0:3448]
	Xvalid=Xtrain[3448:]
	Yvalid=Ytrain[3448:] 
	risk_func,weights=kernel_ridge(Xtraining,Ytraining)
	pred_val=predict(weights, Xtraining, Xvalid)
	benchmark(np.array(pred_val),np.array(Yvalid))

	max_iter=[100,200,500,1000,2000,3000,3448]

	#below code needs to be automated
	training_error[6]=0.27262180974477956
	training_error[0]=0.46113689095127608
	training_error[1]=0.44025522041763343
	training_error[2]=0.46055684454756379
	training_error[3]=0.38399071925754058
	training_error[4]=0.3248259860788863
	training_error[5]=0.30162412993039445

	plt.plot(max_iter,training_error)
	plt.ylabel('Training error')
	plt.xlabel('Iterations')
	plt.show()

      	

