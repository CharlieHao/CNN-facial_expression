# This file construct NN as ANN_dropout_tnsorflow.py shows

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from util import get_data, get_binary_data, error_rate, relu, weights_and_bias_init, y2indicator


class HiddenLayer(object):
	def __init__(self,M1,M2,an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W, b = weights_and_bias_init(M1,M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.params = [self.W, self.b]

	def forward(self,X):
		return tf.nn.relu(tf.matmul(X,self.W)+self.b)

class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=400, batch_sz=100, show_fig=False):
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        decay = np.float32(decay)
        reg = np.float32(reg)
        eps = np.float32(eps)

        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Xtrain, Ytrain = X[:-1000], Y[:-1000]

        # initialize hidden layers
        N, D = Xtrain.shape
        K = len(set(Ytrain))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = weights_and_bias_init(M1, K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        # use tensorflow to build the structure
        inputs = tf.placeholder(tf.float32,shape=(None,D),name='inputs')
        labels = tf.placeholder(tf.int64,shape=(None,),name='labels')

        logits = self.tf_forward(inputs)
        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
        	tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        )
        cost =cost + rcost
        train_op = tf.train.RMSPropOptimizer(learning_rate,decay=decay,momentum=mu).minimize(cost)
        predict_op = self.tf_predict(inputs)

        # combine data and tensorflow structure
        n_batches = int(N/batch_sz)
        costs=[]
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for n in range(n_batches):
                    Xbatch = Xtrain[n*batch_sz:(n*batch_sz+batch_sz)]
                    Ybatch = Ytrain[n*batch_sz:(n*batch_sz+batch_sz)]
                    Ybatch,Yvalid = Ybatch.astype(np.int64),Yvalid.astype(np.int64)

                    session.run(train_op, feed_dict={inputs:Xbatch,labels:Ybatch})

                    if n % 20 == 0:
                        c = session.run(cost, feed_dict={inputs:Xvalid,labels:Yvalid})
                        costs.append(c)

                        p = session.run(predict_op, feed_dict={inputs:Xvalid})
                        err = error_rate(Yvalid, p)
                        print('cost/err at iteration i=%d n=%d is %.3f/%.3f'%(i,n,c,err))
        
        if show_fig:
            plt.plot(costs)
            plt.show()


    def tf_forward(self,X):
        Z = X
        for h in self.hidden_layers:
        	Z = h.forward(Z)
        return tf.matmul(Z,self.W)+self.b

    def tf_predict(self,X):
    	Y = self.tf_forward(X)
    	return tf.argmax(Y,1)
		



def main():
	X,Y = get_data()

	ann = ANN([2000,1000])
	ann.fit(X,Y,show_fig=True)


if __name__ == '__main__':
	main()


