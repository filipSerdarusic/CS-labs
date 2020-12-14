import numpy as np
import matplotlib.pyplot as plt
from data import *

def ReLU(x):
	return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.vstack(e_x.sum(axis=1))


class fcann2():

	def __init__(self, layers, num_iter, param_delta, param_lambda=1e-3):

		self.W1 = np.random.normal(size=(layers[0],layers[1]))
		self.b1 = np.ones(shape=layers[1])
		self.W2 = np.random.normal(size=(layers[1],layers[2]))
		self.b2 = np.zeros(shape=layers[2])

		self.num_iter = num_iter
		self.param_delta = param_delta
		self.param_lambda = param_lambda


	def get_probs(self, X):
		s1 = np.dot(X,self.W1) + self.b1
		h1 = ReLU(s1)
		s2 = np.dot(h1, self.W2) + self.b2
		probs = softmax(s2)
		return probs


	def train(self, X, Y_):
		
		N = X.shape[0]

		for i in range(self.num_iter):

			s1 = np.dot(X,self.W1) + self.b1
			h1 = ReLU(s1)
			s2 = np.dot(h1, self.W2) + self.b2
			probs = softmax(s2)


			reduct = -np.sum(Y_ * np.log(probs), axis=1)
			loss = np.mean(reduct)

			if i%10 == 0:
				print("iteration {}: loss {}".format(i, loss))

			dL = probs
			dL[range(N), np.argmax(Y_, axis=1)] -= 1
			dL /= N

			grad_W2 = np.dot(h1.T, dL)
			grad_b2 = np.sum(dL, axis=0)

			dL_1 = np.dot(dL, self.W2.T)
			dL_relu = dL_1
			dL_relu[h1 <= 0] = 0

			grad_W1 = np.dot(X.T, dL_relu)
			grad_b1 = np.sum(dL_relu, axis=0)

			# regularization
			grad_W2 += self.param_lambda * self.W2
			grad_W1 += self.param_lambda * self.W1

			# weights update
			self.W2 += -self.param_delta * grad_W2
			self.W1 += -self.param_delta * grad_W1
			self.b2 += -self.param_delta * grad_b2
			self.b1 += -self.param_delta * grad_b1

		print("Optimization finished!")

	def classify(self, X):
		probs = self.get_probs(X)
		return np.argmax(probs, axis=1)


if __name__ == "__main__":

	#Initializing random number generators
	np.random.seed(100)

	#Creating data
	X,Y_ = sample_gmm_2d(6, 2, 10)
	Yoh_ = class_to_onehot(Y_).astype(int)

	#Creating a network
	nn = fcann2(layers=[2, 5, 2], num_iter=int(1e5), param_delta=0.05)

	#Learning parameters:
	nn.train(X, Yoh_)

	#Predict classes:
	Y = nn.classify(X)

	#Performance
	accuracy, pr, _ = eval_perf_multi(Y, Y_)
	print("Accuracy: ", accuracy)
	print("Precision / Recall")
	for i in range(len(pr)):
		print("Class {} : {}".format(i, pr[i]))

	#Plotting results
	rect=(np.min(X, axis=0), np.max(X, axis=0))
	graph_surface(nn.classify, rect, offset=0)
	graph_data(X, Y_, Y, special=[])
	plt.show()