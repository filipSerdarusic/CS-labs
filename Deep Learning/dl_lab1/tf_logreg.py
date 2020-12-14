import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from data import *

class TFLogreg:

	def __init__(self, D, C, param_delta=0.05, param_lambda=0.1):
		"""Arguments:
			- D: dimensions of each datapoint
			- C: number of classes
			- param_delta: training step
		"""

		# Defining inputs and groundtruth labels
		self.X = tf.placeholder(tf.float32, [None, D])
		self.Yoh_ = tf.placeholder(tf.float32, [None, C])

		# Defining weights and biases
		self.W = tf.Variable(tf.random_normal([D, C]), name="W")
		self.b = tf.Variable(tf.zeros([C]), name="b")

		# Defining output probabilities
		logit = tf.add(tf.matmul(self.X, self.W), self.b)
		self.probs = tf.nn.softmax(logit)

		# Defining loss function
		reduct = -tf.reduce_sum(self.Yoh_ * tf.log(self.probs), 								reduction_indices=[1])
		self.loss = tf.reduce_mean(reduct)
	
		# Adding regularization term
		self.loss += tf.multiply(0.5*param_lambda, tf.nn.l2_loss(self.W))

		# Defining optmizer
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = param_delta).minimize(self.loss)

		# Initializing global variables
		self.init = tf.initializers.global_variables()
		self.sess = tf.Session()
	

	def train(self, X, Yoh_, epochs):
		"""Arguments:
			- X: actual datapoints [NxD]
			- Yoh_: one-hot encoded labels [NxC]
			- epochs: number of epochs
		"""

		self.sess.run(self.init)
		for i in range(epochs):
			
			feed_dict = {
						self.X:X,
						self.Yoh_:Yoh_
						}

			self.sess.run(self.optimizer, feed_dict=feed_dict)

			loss = self.sess.run(self.loss, feed_dict=feed_dict)

			if i % 10 == 0:
				print("iteration {}: loss {}".format(i, loss))

		print("Optimization finished!")


	def eval(self, X):
		"""Arguments:
			- X: actual datapoints [NxD]
			Returns: predicted class probabilites [NxC]
		"""
		probs = self.sess.run(self.probs, feed_dict={self.X:X})
		return probs
	

	def eval_class(self, X):
		"""Arguments:
			- X: actual datapoints [NxD]
			Returns: predicted classes [N]
		"""
		probs = self.eval(X)
		return probs.argmax(axis=1)


if __name__ == "__main__":

	#Initialize random number generators
	np.random.seed(100)
	tf.set_random_seed(100)

	#Instatiate data X and labels Yoh_
	X,Y_ = sample_gauss_2d(3, 100)
	Yoh_ = class_to_onehot(Y_)

	#Build a graph:
	tflr = TFLogreg(X.shape[1], Yoh_.shape[1], param_delta=0.1, param_lambda=0.01)

	#Learn parameters:
	tflr.train(X, Yoh_, epochs=5000)

	#Get probabilites on train set:
	probs = tflr.eval(X)

	# Predicted classes
	Y = tflr.eval_class(X)

	#Print performance (precision and recall for each class)

	#Plot results
	rect=(np.min(X, axis=0), np.max(X, axis=0))
	graph_surface(tflr.eval_class, rect, offset=0)
	graph_data(X, Y_, Y, special=[])
	plt.show()