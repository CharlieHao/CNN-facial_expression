import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from util import get_binary_data, sigmoid, relu, sigmoid_cost, error_rate

class ANN_sigmoid(object):
	def __init__(self,M):
		# M is the number of the hidden units, and is the attribute
		self.M = M

	def fit(self,X,T,learning_rate = 5*10e-7,reg =1.0,epoch=10000,fig_show=False):
		X,T = shuffle(X,T)
		X_train = X[:-1000]
		T_train = T[:-1000]
		X_valid = X[-1000:]
		T_valid = T[-1000:]

		N,D = X_train.shape
		self.W1 = np.random.randn(D,self.M)/np.sqrt(D)
		self.b1 = np.zeros(self.M)
		self.W2 = np.random.randn(self.M)/np.sqrt(self.M)
		self.b2 = 0

		best_validation_error = 1
		costs = []
		for n in range(epoch):
			# forwardpropogation
			Y,Z= self.forwardprop(X_train)

			#backpropogation process and Gradient descent
			Y_T = Y-T_train
			self.W2 -= learning_rate*(Z.T.dot(Y_T)+reg*self.W2)
			self.b2 -= learning_rate*((Y_T).sum(axis=0)+reg*self.b2)
			
			dZ = np.outer(Y_T,self.W2)*(1-Z*Z)
			self.W1 -= learning_rate*(X_train.T.dot(dZ)+reg*self.W1)
			self.b1 -= learning_rate*(dZ.sum()+reg*self.b1)

			if n%20 == 0:
				Y_valid, _= self.forwardprop(X_valid)
				cost = sigmoid_cost(T_valid,Y_valid)
				costs.append(cost)
				#must use print to give a feedback
				er = error_rate(T_valid,np.round(Y_valid))
				print(n,'cost:',cost,'error rate',er)
				if er<best_validation_error:
					best_validation_error = er
		print('Best validation error:',best_validation_error)

		if fig_show:
			plt.plot(costs)
			plt.show()

	def forwardprop(self,X):
		Z = np.tanh(X.dot(self.W1)+self.b1)
		return sigmoid(Z.dot(self.W2)+self.b2),Z

	def predict(self,X):
		Y,_ = self.forwardprop(X)
		return np.round(Y)

	def score(self,X,T):
		Y= self.predict(X)
		return 1-error_rate(T,Y)



def main():
	X,Y= get_binary_data()
	#from util package disussion, there is an inbalance of class
	#need to expand class 1
	X0,Y0 = X[Y==0],Y[Y==0]
	X1,Y1 = X[Y==1],Y[Y==1]
	X1 = np.repeat(X1,9,axis=0)
	X = np.vstack([X0,X1])
	Y = np.concatenate([Y0,[1]*len(X1)])

	model = ANN_sigmoid(100)
	model.fit(X,Y,fig_show=True)


if __name__ == '__main__':
	main()



















