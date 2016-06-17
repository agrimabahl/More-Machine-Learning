%pylab inline

import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def main():
	tr = scipy.io.loadmat('housing_data.mat')


	train=tr["Xtrain"]
	train_labels=tr["Ytrain"]
	test=tr["Xvalidate"]
	validate_labels=tr["Yvalidate"]

	train_label=[0]*len(train_labels)
	validate_label=[0]*len(validate_labels)

	#flatten train labels
	for i in range(0,len(train_labels)):
    	train_label[i]=tr["Ytrain"][i][0]

    #flatten validate labels
	for i in range(0,len(validate_labels)):
    	validate_label[i]=tr["Yvalidate"][i][0]

    alpha=mean(train_label)
	train=np.insert(train, 8, 1, axis=1)
	lambda_1=0.2
	pseudo_inv=np.linalg.inv(train.transpose().dot(train)+lambda_1*np.identity(9))
	w=pseudo_inv.dot(train.transpose()).dot(train_label)
	y_bar=test.dot(np.matrix(w).transpose())
	w = w.reshape((8,1))
	rss=np.sum((y_bar-validate_label)**2)

	#plotting the predicted weights (w's)
	plt.plot([0,1,2,3,4,5,6,7,8],[w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]])
	plt.show




# cross-validation code below

	for i in range(0,10):
    	validation = consolidated[i*subset_size:(i+1)*subset_size]
    	labelv = consolidated_labels[i*subset_size:(i+1)*subset_size]
    	training = np.concatenate((consolidated[:i*subset_size],consolidated[(i+1)*subset_size:]))
    	labelt = np.concatenate((consolidated_labels[:i*subset_size],consolidated_labels[(i+1)*subset_size:]))
    	lambda_1=pow(0.2,i-5)
    	pseudo_inv=np.linalg.inv(training.transpose().dot(training)+lambda_1*np.identity(9))
    	w=pseudo_inv.dot(training.transpose()).dot(labelt)
    	w = w.reshape((9,1))
    	X=training.dot(w)-labelt
    	#loss_function=X.transpose().dot(X)+lamda_1*w.transpose().dot(w)
    	y_bar=validation.dot(np.matrix(w))
    	rss=np.sum((y_bar-labelv)**2)
    	print (rss)
/	