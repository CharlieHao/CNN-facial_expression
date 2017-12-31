#This package involves these functions may be uesrd in 
#the process of building CNN and other networks

import numpy as np 
import pandas as pd 

def get_data(balance_class_one = True):
	#images are 48 x 48 = 2304 sizes vector
	X = []
	Y = []
	title_row = True
	for line in open('fer2013.csv'):
		if title_row == True:
			title_row = False
		else:
			row = line.split(',')
			y = int(row[0])
			x = [int(n) for n in row[1].split()]
			Y.append(y)
			X.append(x)

	X,Y = np.array(X)/255.0,np.array(Y)

	#based on preprocess of data, there is an inbalance class: class 1
	if balance_class_one == True:
		X0, Y0 = X[Y!=1,:],Y[Y!=1]
		X1 = X[Y==1,:]
		X1 = np.repeat(X1,9,axis = 0)
		X = np.vstack((X0,X1))
		Y = np.concatenate((Y0,[1]*len(X1)))

	return X,Y

def get_image_data():
	X,Y = get_data()
	N,D = X.shape
	d = int(np.sqrt(D))
	X = X.reshape(N,1,d,d)
	return X,Y

def get_binary_data():
	# this function is used to get the binary data of the first two class
	X =[]
	Y =[]
	first_row = True
	for line in open('fer2013.csv'):
		if first_row == True:
			first_row = False
		else:
			row = line.split(',')
			y = int(row[0])
			if y==0 or y==1:
				x = [int(n) for n in row[1].split()]
				Y.append(y)
				X.append(x)
	return np.array(X)/255.0,np.array(Y)

def Cross_validation(model,X,Y,K=5):
	X,Y = shuffle(X,Y)
	errors = []
	size = len(Y)/K
	for k in range(K):
		X_train = np.vstack([X[:k*size,:]],X[(k+1)*size:,:])
		y_train = np.concatenate((X[:k*size,:],X[(k+1)*size:,:]))
		x_test = X[k*size:(k+1)*size,:]
		y_test = Y[k*size:(k+1)*size,:]

		model.fit(X_train,y_train)
		error = model.score(X_test,Y_test)
		errors.append(error)
	return np.mean(errors)


def weights_and_bias_init(D1,D2):
	# D1 and D2 are the dimentionality of input and output
	# parameters should be inependent of the number of parameters
	W = np.random.randn(D1,D2)/np.sqrt(D1)
	b = np.zeros(D2)
	return W.astype(np.float32),b.astype(np.float32)

def softmax(a):
	A = np.exp(a)
	return A/A.sum(axis=1,keepdims=True)

def sigmoid(a):
	return 1/(1+np.exp(-a))

def relu(x):
	return x*(x>0)

def init_filter(shape, poolsz):
	#this function is used in the concolutional nueral network, this is for theano
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)

def sigmoid_cost(T,Y):
	return -(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()

def cost2(T,Y):
	#T should be the indicator matrix for crossentropy, but there: T is the target matrix
	N = len(Y)
	return -np.log(Y[np.arange(N),T]).sum()

def cost(T,Y):
	# T is the indicator matrix
	return -(T*np.log(Y)).sum()

def error_rate(target,prediction):
	return np.mean(target!=prediction)

def predict(Y_given_X):
	return np.argmax(Y_given_X,axis=1)

def y2indicator(Y):
	N = len(Y)
	K = len(set(Y))
	T = np.zeros([N,K])
	for n in range(N):
		T[n,Y[n]] = 1
	return T

