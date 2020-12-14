import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import *


class TFDeep():

	def __init__(self, layers, learning_rate=0.1, param_lambda=0.1):

		self.W = []
		self.b = []

		# Constructing layers
		i = 1
		while i < len(layers):
			self.W.append(tf.Variable(tf.truncated_normal(shape=[layers[i-1], layers[i]], stddev=0.35)))
			self.b.append(tf.Variable(tf.zeros([layers[i]])))
			i += 1

		# Defining inputs and outputs
		self.X = tf.placeholder(tf.float32, [None, layers[0]])
		self.Yoh_ = tf.placeholder(tf.float32, [None, layers[-1]])

		# Initializing variables
		self.sess = tf.Session()

		# Defining network outputs
		h = self.X
		for w, b in zip(self.W, self.b):
			s = tf.matmul(h, w) + b
			if len(layers) > 2 and (self.W.index(w)+2) != len(layers):
				h = tf.nn.relu(s)
			else:
				h = s
		self.probs = tf.nn.softmax(h)

		# Defining loss
		"""
		reduct = -tf.reduce_sum(self.Yoh_ * tf.log(self.probs), 								reduction_indices=[1])
		self.loss = tf.reduce_mean(reduct)
		"""
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(Yoh_, h))
	
		# Adding L2 regularization term
		self.reg_loss = 0
		for w in self.W:
			self.reg_loss += 0.5*param_lambda*tf.nn.l2_loss(w)
		self.loss += self.reg_loss

		# Defining optmizer
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)

	def train(self, X, Yoh_, epochs):
		"""Arguments:
			- X: actual datapoints [NxD]
			- Yoh_: one-hot encoded labels [NxC]
			- epochs: number of epochs
		"""
		init = tf.initializers.global_variables()
		self.sess.run(init)

		for i in range(epochs):

			feed_dict = {
						self.X:X,
						self.Yoh_:Yoh_
						}

			self.sess.run(self.train_op, feed_dict=feed_dict)
			loss = self.sess.run(self.loss, feed_dict=feed_dict)

			if i % 10 == 0:
				print("iteration {}: loss {}".format(i, loss))

		print("Optimization finished!")
	
	def eval(self, X):
		"""Arguments:
			- X: actual datapoints [NxD]
			Returns: predicted class probabilites [NxC]
		"""
		probs = self.sess.run(self.probs, feed_dict={self.X : X})
		return probs

	def eval_class(self, X):
		"""Arguments:
			- X: actual datapoints [NxD]
			Returns: predicted class [N]
		"""
		probs = self.eval(X)
		return probs.argmax(axis=1)
	
	def count_params(self):
		total_params = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape()
			variable_params = 1
			for dim in shape:
				variable_params *= dim.value
			total_params += variable_params
		return total_params

if __name__ == "__main__":

	#Initializing random number generators
	np.random.seed(100)
	tf.set_random_seed(100)

	#Creating data
	X,Y_ = sample_gmm_2d(6, 2, 10)
	Yoh_ = class_to_onehot(Y_)

	#Building the graph
	layers = [2, 10, 10, 2]
	deep = TFDeep(layers=layers, learning_rate=0.1, param_lambda=1e-4)

	#Number of parameters
	num_of_params = deep.count_params()
	print("Number of parameters: ", num_of_params)

	#Learning
	deep.train(X, Yoh_, epochs=int(1e4))

	#Probabilites on train set:
	probs = deep.eval(X)

	# redicted classes
	Y = deep.eval_class(X)

	#Print performance (precision and recall for each class)
	accuracy, pr, M = eval_perf_multi(deep.eval_class(X), Y_)
	print("Accuracy: ", accuracy)
	print("Precision / Recall")
	for i in range(len(pr)):
		print("Class {} : {}".format(i, pr[i]))
	print("Confusion Matrix:\n ", M)

	#Plot results
	rect=(np.min(X, axis=0), np.max(X, axis=0))
	graph_surface(deep.eval_class, rect, offset=0)
	graph_data(X, Y_, Y, special=[])
	plt.show()
