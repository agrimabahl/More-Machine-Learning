import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

def logistic(x):
    return 1/(1 + math.exp(-1*x))

def risk(w,X,Y):
    risk_f=0
    for i in range(0,5172):
        wx=w.transpose().dot(X[i])
        sig=logistic(wx)
        if sig==1:
            sig=0.99999995
        if sig==0:
            sig=0.00000005
        risk_f+=-(Y[i]*np.log(sig)+(1-Y[i])*np.log(1-sig))
    return risk_f
def d(w,X,Y):
    diff=[0]*33
    for i in range(0,5172):
        diff+=X[i]*(Y[i]-logistic(X[i].transpose().dot(w)))
    return diff    

def batchgradient(X,Y):
    max_iter=3000
    eps = 0.001
    w=np.array([0] * 33, 'float')
    w[32]=1
    risk_fn=np.zeros(max_iter/50,dtype='float')
    for i in range(max_iter):
        if i%50 ==0:
            j=int(i/50)
            risk_fn[j]=risk(w,X,Y)
        w=w+eps*d(w,X,Y)
        print("at iter",i)
        print("Risk fn value=",risk_fn[j])
        print("weights are",w)
    return (w,risk_fn)

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


	w=np.zeros(32,dtype='float')

	w=np.array([0] * 33, 'float')

	weights,risk_function=batchgradient(Xtrain, Ytrain)
	Xtest = np.insert(Xtest, 32, values=1, axis=1)

	#Pre-Pro (i) scaling the columns individually
	Xtrain=np.asfarray(Xtrain)
	
	for i in range(0,32):
		Xtrain[:,i]=preprocessing.scale(Xtrain[:,i])

    #Pre-Pro (ii)
	#pre-processing method 2 : log(x+0.1)
	'''Xtrain2=np.asfarray(Xtrain)
	for i in range(0,5172):
    	for j in range(0,32):
        	Xtrain2[i][j]=np.log(Xtrain[i][j]+0.1)

	#Pre-Pro (iii)
	#pre-processing method 3: Binarize i.e. if x>0, x=1 else x=0

	for i in range(0,5172):
    	for j in range(0,32):
        	if Xtrain[i][j]>0:
            	Xtrain[i][j]=1
        	else:
            	Xtrain[i][j]=0
            	'''
    #adding one column for bias
	Xtrain = np.insert(Xtrain, 32, values=1, axis=1)
	weights,risk_function=batchgradient(Xtrain, Ytrain)
	Xtest=tr["test_data"]

	pred_val=predict(weights, Xtraining, Xvalid)
	benchmark(np.array(pred_val),np.array(Yvalid))


	x = [i*50 for i in range(len(risk_function3))]
	plt.plot(x,risk_function3)
	#plt.ylabel('some numbers')
	plt.show()


            