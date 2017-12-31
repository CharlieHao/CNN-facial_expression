import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from util import get_image_data, error_rate, weights_and_bias_init, y2indicator
from ann_tensorflow import HiddenLayer

# differences from Theano:
# image dimensions are expected to be: N x width x height x color
# filter shapes are expected to be: filter width x filter height x input feature maps x output feature maps

def init_filter(shape, poolsz):
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	return w.astype(np.float32)


class ConvpoolLayer(object):
	def __init__(self,mi,mo,fw=5,fh=5,poolsz=(2,2)):
		sz = (fw, fh, mi, mo)
		W0 = init_filter(sz,poolsz)
		self.W = tf.Variable(W0)
		b0 = np.zeros(mo,dtype=np.float32)
		self.b = tf.Variable(b0)
		self.poolsz = poolsz
		self.params = [self.W,self.b]

	def forward(self,X):
		conv_out = tf.nn.conv2d(X,self.W,strides=[1,1,1,1],padding='SAME')
		conv_out = tf.nn.bias_add(conv_out,self.b)
		p1,p2 = self.poolsz
		pool_out = tf.nn.max_pool(
			conv_out,
			ksize=(1,p1,p2,1),
			strides=(1,p1,p2,1),
			padding="SAME")
		return tf.tanh(pool_out)

class CNN(object):
	def __init__(self,convpool_layer_sizes,hidden_layer_sizes):
		# conv_pool_size: (mo,fw,fh), hedden_layer_size: M_out
		self.convpool_layer_sizes = convpool_layer_sizes
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, Y, lr=10e-4, mu=0.99, reg=10e-4, decay=0.99999, eps=10e-3, batch_sz=30, epochs=4, show_fig=True):
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
		Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
		Yvalid_ind = y2indicator(Yvalid).astype(np.float32)

		# step 2: initialize weights in convpool layers and mlp layers
		# convpool use padding='valid', convpool initialization
		N,width,height,c = Xtrain.shape
		mi = c 
		outw = width
		outh = height
		self.convpool_layers = []
		for mo,fw,fh in self.convpool_layer_sizes:
			h = ConvpoolLayer(mi,mo,fw,fh)
			self.convpool_layers.append(h)
			outw = outw//2
			outh = outh//2
			mi = mo

		# mlp initialization
		K = len(set(Ytrain))
		M1 = self.convpool_layer_sizes[-1][0]*outw*outh
		count = 0
		self.hidden_layers = []
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1,M2,count)
			self.hidden_layers.append(h)
			count += 1
			M1 = M2

		# the output layer
		W,b = weights_and_bias_init(M1,K)
		self.W = tf.Variable(W,'W_logreg')
		self.b = tf.Variable(b,'b_logreg')

		# collect all parameters matrix as a list
		self.params = [self.W, self.b]
		for h in self.convpool_layers:
			self.params += h.params
		for h in self.hidden_layers:
			self.params += h.params

		# step3: tensorflow structure, cost expression and train, predict operation
		tfX = tf.placeholder(tf.float32,shape=(None,width,height,c),name='X')
		tfT = tf.placeholder(tf.float32,shape=(None,K),name='T')
		pY = self.tf_forward(tfX)

		rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
		cost = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
				logits=pY,
				labels=tfT
			)
		)+rcost

		predict_op = self.tf_predict(tfX)
		train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

		# combine data and tensorflow structure
		n_batches = N//batch_sz
		costs=[]
		init = tf.global_variables_initializer()

		with tf.Session() as session:
			session.run(init)
			for i in range(epochs):
				Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
				for n in range(n_batches):
					Xbatch = Xtrain[n*batch_sz:(n*batch_sz+batch_sz)]
					Ybatch = Ytrain_ind[n*batch_sz:(n*batch_sz+batch_sz)]

					session.run(train_op, feed_dict={tfX:Xbatch,tfT:Ybatch})

					if n % 20 == 0:
						c = session.run(cost, feed_dict={tfX:Xvalid,tfT:Yvalid_ind})
						costs.append(c)

						p = session.run(predict_op, feed_dict={tfX:Xvalid})
						err = error_rate(Yvalid, p)
						print('cost/err at iteration i=%d n=%d is %.3f/%.3f'%(i,n,c,err))

		if show_fig:
			plt.plot(costs)
			plt.show()

	def tf_forward(self,X):
		Z = X
		for h in self.convpool_layers:
			Z = h.forward(Z)
		Z_shape = Z.get_shape().as_list()
		Z = tf.reshape(Z,[-1,np.prod(Z_shape[1:])])
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return tf.matmul(Z,self.W)+self.b

	def tf_predict(self,X):
		Y = self.tf_forward(X)
		return tf.argmax(Y,1)


def main():
	X,Y = get_image_data()
	X = X.transpose((0,2,3,1))
	model = CNN(
		convpool_layer_sizes=[(20,5,5),(20,5,5)],
		hidden_layer_sizes=[500,300]
	)
	model.fit(X,Y)

if __name__ == '__main__':
	main()









