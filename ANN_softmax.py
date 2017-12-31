import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from util import get_data, softmax, relu, cost2, error_rate,y2indicator

class ANN_softmax(object):
	def __init__(self,M):
		self.M = M

	def fit(self,X,T,learning_rate=10e-7,reg=10e-7,epochs=10000,show_fig=False):
		X,T = shuffle(X,T)
		X_train, T_train = X[:-1000],T[:-1000]
		X_valid,T_valid = X[-1000:],T[-1000:]

		N,D = X_train.shape
		K = len(set(T_train))

		#initialize parameters
		self.W1 = np.random.randn(D,self.M)/np.sqrt(D)
		self.b1 = np.zeros(self.M)
		self.W2 = np.random.randn(self.M,K)/np.sqrt(self.M)
		self.b2 = np.zeros(K)

		costs =[]
		best_validation_error = 1
		for n in range(epochs):
			#forwardpropogation process
			Y,Z = self.forwardprop(X_train)

			#Gradient Descent
			T_train_ind = y2indicator(T_train)
			Y_T = Y-T_train_ind
			self.W2 -= learning_rate*(Z.T.dot(Y_T)+reg*self.W2)
			self.b2 -= learning_rate*(Y_T.sum(axis=0)+reg*self.b2)
			
			dZ = Y_T.dot(self.W2.T)*(1-Z*Z)
			self.W1 = learning_rate*(X_train.T.dot(dZ)+reg*self.W1)
			self.b1 = learning_rate*(dZ.sum(axis=0)+reg*self.b1)

			#representation of validation cost and error rate
			if n%10 == 0:
				Y_valid,_ = self.forwardprop(X_valid)
				cost = cost2(T_valid,Y_valid)
				costs.append(cost)
				er = error_rate(T_valid,np.argmax(Y_valid,axis=1))
				print(n,'cost:',cost,'error',er)
				if er < best_validation_error:
					best_validation_error=er
		print('Best validation error:',best_validation_error)

		if show_fig:
			plt.plot(costs)
			plt.title('cross entropy loss')
			plt.show()

	def forwardprop(self,X):
		Z = np.tanh(X.dot(self.W1)+self.b1)
		return softmax(Z.dot(self.W2)+self.b2),Z

	def predict(self,X):
		Y = self.forwardprop(X)
		return np.argmax(Y,axis=1)

	def score(self,X,T):
		Y = self.predict(X)
		return 1- error_rate(T,Y)


def main():
	X,T = get_data()

	model = ANN_softmax(200)
	model.fit(X,T,show_fig=True)
	print(model.score(X,T))

if __name__ == '__main__':
	main()












