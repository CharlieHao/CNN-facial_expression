import numpy as np 
import theano
import theano.tensor as T 
import matplotlib.pyplot as plt 

from scipy.utils import shuffle
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from util import get_image_data, error_rate, weights_and_bias_init, init_filter
from ann_theano import HiddenLayer 

class ConvpoolLayer(object):
	def __init__(self,mi,mo,fw=5,fh=5,poolsz=(2,2)):
		# mi: input feature maps, mo: output feature maps, 
		#fw: filter width, fh: filter height
		sz = (mi, mo, fw, fh)
		W0 = init_filter(sz,poolsz)
		self.W = theano.shared(W0)
		b0 = np.zeros(mo, dtype=np.float32)
		self.b = theano.shared(b0)
		self.poolsz = poolsz
		self.params = [self.W,self.b]

	def forward(self,X):
		conv_out = conv2d(X,self.W)
		pool_out = downsample.max_pool_2d(
			input=conv_out,
			ds=self.poolsz,
			ignore_border=True
		)
		return T.tanh(pool_out+self.b.dimshuffle('x',0,'x','x'))

class CNN(object):
	def __init__(self,conv_pool_size,hidden_layer_size):
		# conv_pool_size: (mo,fw,fh), hedden_layer_size: M_out
		self.conv_pool_size = conv_pool_size
		self.hidden_layer_size = hidden_layer_size

	def fit(self, X, Y, lr=10e-5, mu=0.99, reg=10e-7, decay=0.99999, eps=10e-3, batch_sz=30, epochs=100, show_fig=True):
		# step 1: process parameters to suitble type and preprocess input data
		lr = np.float32(lr)
		mu = np.float32(mu)
		reg = np.float32(reg)
		decay = np.float32(decay)
		eps = np.float32(eps)

		# make a validation set
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]
		Xtrain, Ytrain = X[:-1000], Y[:-1000]

		# step 2: initialize weights in convpool layers and mlp layers
		# convpool use padding='valid', convpool initialization
		N,c,width,height = Xtrain.shape
		mi = c 
		outw = width
		outh = height
		self.convpool_layers = []
		for mo,fw,fh in self.conv_pool_size:
			h = ConvpoolLayer(mi,mo,fw,fh)
			self.convpool_layers.append(h)
			outw = (outw - fw + 1)//2
			outh = (outh - fh + 1)//2
			mi = mo

		# mlp initialization
		K = len(Ytrain)
		M1 = self.conv_pool_size[-1][0]*outw*outh
		count = 0
		self.hidden_layers = []
		for M2 in self.hidden_layer_size:
			h = HiddenLayer(M1,M2,count)
			self.hidden_layers.append(h)
			count += 1
			M1 = M2

		# the output layer
		W,b = weights_and_bias_init(M1,K)
		self.W = theano.shared(W,'W_logreg')
		self.b = theano.shared(b,'b_logreg')

		# collect all parameters matrix as a list
		self.params = [self.W, self.b]
		for h in self.convpool_layers:
			self.params += h.params
		for h in self.hidden_layers:
			self.params += h.params

		# step 3: theano structure and cost, prediction, and updates expression
		# initialize: (momentum and RMSprop)
		dparams = [theano.shared(np.zeros(p.get_value().shape,dtype=np.float32)) for p in self.params]
		cache = [theano.shared(np.zeros(p.get_value().shape,dtype=np.float32)) for p in self.params]

		thX = T.tensor4('X',dtype='float')
		thT = T.ivector('T')
		Y = self.th_forward(thX)

		rcost = reg*T.sum((p*p).sum() for p in self.params)
		cost = -T.mean(T.log(Y[T.arrange(thT.shape[0]),thT]))+rcost
		prediction = self.th_predict(thX)

		self.predict_op = theano.function(inputs=[thX], outputs=prediction)
		cost_predict_op = theano.function(inputs=[thX, thT], outputs=[cost, prediction])

		updates = [
		    (p, p + mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
		] + [
		    (dp, mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
		]
		train_op = theano.function(
			inputs=[thX, thT],
			updates=updates
		)

		n_batches = N // batch_sz
		costs = []
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
				Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

				train_op(Xbatch, Ybatch)

				if j % 20 == 0:
					c, p = cost_predict_op(Xvalid, Yvalid)
					costs.append(c)
					e = error_rate(Yvalid, p)
					print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

		if show_fig:
			plt.plot(costs)
			plt.show()

	def th_forward(self,X):
		Z = X
		for h in self.convpool_layers:
			Z = h.forward(Z)
		Z = Z.flatten(ndim=2)
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return T.nnet.softmax(Z.dot(self.W)+self.b)

	def th_predict(self,X):
		Y = self.predict_op(X)
		return T.argmax(Y,axis=1)

def main():
	X,Y = get_image_data()
	model = CNN(
		conv_pool_size=[(20,5,5),(20,5,5)],
		hidden_layer_size=[500,300]
	)
	model.fit(X,Y)

if __name__ == '__main__':
	main()





















