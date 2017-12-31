import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from util import get_data, softmax, relu, cost, error_rate, y2indicator

class LogisticModel(object):
	def __init__(self):
		pass

	def fit(self,X,T,learning_rate=10e-8,reg=10e-12,epochs=10000,show_fig=False):
		X,T = shuffle(X,T)
		X_train,T_train = X[:-1000],T[:-1000]
		X_valid,T_valid = X[-1000:],T[-1000:]

		N,D = X_train.shape
		K =len(set(T_train))
		T_train_ind = y2indicator(T_train)

		#initialize parameter: W need independence to number of para
		self.W = np.random.randn(D,K)/np.sqrt(D+K)
		self.b = np.zeros(K)

		costs = []
		best_validation_error = 1
		for n in range(epochs):
			# forwardpropogation process
			Y = self.forwardprop(X_train)

			#Gradient descent
			Y_T = Y-T_train_ind
			self.W -= learning_rate*(X_train.T.dot(Y_T)+reg*self.W)
			self.b -= learning_rate*(Y_T.sum(axis=0)+reg*self.b)

			#presentation
			if n%10 == 0:
				Y_valid = self.forwardprop(X_valid)
				T_valid_ind = y2indicator(T_valid)
				c = cost(T_valid_ind,Y_valid)
				costs.append(c)
				er = error_rate(T_valid,self.predict(X_valid))
				print(n,'cost',c,'error',er)
				if er < best_validation_error:
					best_validation_error = er

		print('Best validation error', best_validation_error)

		if show_fig:
			plt.plot(costs)
			plt.title('cross entropy loss')
			plt.show()

	def forwardprop(self,X):
		return softmax(X.dot(self.W)+self.b)

	def predict(self,X):
		Y = self.forwardprop(X)
		return np.argmax(Y,axis=1)

	def score(self,X,T):
		Y = self.predict(X)
		return 1-error_rate(T,Y)


def main():
	X,Y = get_data()
	
	model = LogisticModel()
	model.fit(X,Y,show_fig=True)
	print(model.score(X,Y))

if __name__ == '__main__':
	main()











