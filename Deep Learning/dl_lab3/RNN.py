from dataset import *
import numpy as np
import matplotlib.pyplot as plt

batch_size = 10
seq_lenght = 20
input_dim = 1

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / np.vstack(e_x.sum(axis=1))

class RNN:

	def __init__(self, vocab_size, sequence_length=20,
				 hidden_size=200, lr=1e-2):

		self.hidden_size = hidden_size
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.lr = lr

		# input projection
		self.U = np.random.randn(vocab_size, hidden_size)

		# hidden-to-hidden projection
		self.W = 1e-2 * np.random.randn(hidden_size, hidden_size)

		# input bias
		self.b = np.zeros((1, hidden_size))

		# output projection
		self.V = np.random.randn(hidden_size, vocab_size)

		# output bias
		self.c = np.zeros((1, vocab_size))

		# memory of past gradients - rolling sum of squares for Adagrad
		self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)

		self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

	
	def rnn_step_forward(self, x, h_prev, U, W, b):

		# A single time step forward of a recurrent neural network with a 
		# hyperbolic tangent nonlinearity.

		# x - input data (minibatch size x input dimension)
		# h_prev - previous hidden state (minibatch size x hidden size)
		# U - input projection matrix (input dimension x hidden size)
		# W - hidden to hidden projection matrix (hidden size x hidden size)
		# b - bias of shape (hidden size x 1)

		# Maybe x will need reshaping -> x.reshape(-1,1)

		h_current = np.tanh(np.dot(h_prev, W) + np.dot(x, U) + b)

		cache = (h_prev, h_current, x, U, W, b)
		
		# return the new hidden state and a tuple of values needed for the backward step

		return h_current, cache


	def rnn_forward(self, x, h0, U, W, b):

		# Full unroll forward of the recurrent neural network with a 
		# hyperbolic tangent nonlinearity

		# x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
		# h0 - initial hidden state (minibatch size x hidden size)
		# U - input projection matrix (input dimension x hidden size)
		# W - hidden to hidden projection matrix (hidden size x hidden size)
		# b - bias of shape (hidden size x 1)
		
		batch_len, seq_len, input_dim = x.shape

		cache = []
		h = np.zeros((batch_len, seq_len, self.hidden_size))

		h_current = h0
		for t in range(self.sequence_length):

			h_current, cache_current = self.rnn_step_forward(x[:,t,:], 														h_current, U, W, b)
			h[:, t, :] = h_current
			cache.append(cache_current)

		# return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

		return h, cache


	def rnn_step_backward(self, grad_next, cache):
		# A single time step backward of a recurrent neural network with a 
		# hyperbolic tangent nonlinearity.

		# grad_next - upstream gradient of the loss with respect to the next hidden state and current output
		# cache - cached information from the forward pass

		h_prev, h_current, x, U, W, b = cache
		

		dtanh = (1 - h_current**2) * grad_next
		dh_prev = np.dot(dtanh, W)
	
		# Gradient of loss with respect to U
		dU = np.dot(x.T, dtanh)
		dU = np.clip(dU, -5, 5)

		# Gradient of loss with respect to W
		dW = np.dot(dtanh.T, h_prev)
		dW = np.clip(dW, -5, 5)

		# Gradient of loss with respect to b
		db = np.sum(dtanh, axis=0).reshape(1,-1)
		db = np.clip(db, -5, 5)
		
		# compute and return gradients with respect to each parameter
		# HINT: you can use the chain rule to compute the derivative of the
		# hyperbolic tangent function and use it to compute the gradient
		# with respect to the remaining parameters

		return dh_prev, dU, dW, db


	def rnn_backward(self, dh, cache):
		# Full unroll forward of the recurrent neural network with a 
		# hyperbolic tangent nonlinearity
		
		dU, dW, db = None, None, None
		
		# Cache from the full forward pass
		h_prev, h_current, x, U, W, b = cache[-1]

		# Initialize gradients (dims?)
		dU = np.zeros_like(U)
		dW = np.zeros_like(W)
		db = np.zeros_like(b)

		#dh0 = np.zeros_like(h_current)
		dh_prev = np.zeros_like(h_prev)
		
		# Looping through time steps starting from the last
		for t in reversed(range(self.sequence_length)):
			
			dh_prev, dUt, dWt, dbt = self.rnn_step_backward(dh_prev, cache[t])

			# Incrementing global derivatives with respect to parameters by adding their derivative at time-step
			dU += dUt
			dW += dWt
			db += dbt

		# compute and return gradients with respect to each parameter
		# for the whole time series.
		# Why are we not computing the gradient with respect to inputs (x)?

		return dU, dW, db


	def output(self, h, V, c):
		# Calculate the output probabilities of the network
		return np.dot(h, V) + c


	def loss(self, y_, y):
		# y - (batch_size x vocab_size)
		return -np.sum(y_*np.log(y))/len(y)


	def output_loss_and_grads(self, h, V, c, y):
		# Calculate the loss of the network for each of the outputs
		
		# h - hidden states of the network for each timestep. 
		#     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
		# V - the output projection matrix of dimension hidden size x vocabulary size
		# c - the output bias of dimension vocabulary size x 1
		# y - the true class distribution - a tensor of dimension 
		#     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
		#     passing the argument. A fast way to create a one-hot vector from
		#     an id could be s omething like the following code:

		#   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
		#   y[batch_id][timestep][batch_y[timestep]] = 1

		#     where y might be a list or a dictionary.

		loss, dh, dV, dc = None, None, None, None

		dh = np.zeros_like(h)
		dh_prev = np.zeros_like(h[:,0,:])

		loss = np.zeros(self.sequence_length)

		for t in range(self.sequence_length):

			o = self.output(h[:,t,:], V, c)
			probs = softmax(o)
			loss[t] = self.loss(y[:,t,:], probs)

			do = probs - y[:,t,:]
			dh[:,t,:] = np.dot(do, V.T) + dh_prev
			dh_prev = dh[:,t,:]

			if t == self.sequence_length - 1:
				dV = np.dot(h[:,t,:].T,do)
				dc = np.sum(do,axis=0).reshape(1,-1)

		# calculate the output (o) - unnormalized log probabilities of classes
		# calculate yhat - softmax of the output
		# calculate the cross-entropy loss
		# calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
		# calculate the gradients with respect to the output parameters V and c
		# calculate the gradients with respect to the hidden layer h

		return loss, dh, dV, dc


	def update(self, dU, dW, db, dV, dc):
		# update memory matrices
		# perform the Adagrad update of parameters

		delta = 1e-7
		self.memory_U += np.square(dU)
		self.memory_W += np.square(dW)
		self.memory_b += np.square(db)
		self.memory_V += np.square(dV)
		self.memory_c += np.square(dc)
		
		self.U += -(self.lr/(delta + np.sqrt(self.memory_U))) * dU
		self.W += -(self.lr/(delta + np.sqrt(self.memory_W))) * dW
		self.b += -(self.lr/(delta + np.sqrt(self.memory_b))) * db
		self.V += -(self.lr/(delta + np.sqrt(self.memory_V))) * dV
		self.c += -(self.lr/(delta + np.sqrt(self.memory_c))) * dc


	def step(self, h0, x_oh, y_oh):
		# Forward and backward pass of RNN
		# Returns the loss and hidden state in the last step

		h, cache = self.rnn_forward(x_oh, h0, self.U, self.W, self.b)
		loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y_oh)
		dU, dW, db = self.rnn_backward(dh, cache)
		self.update(dU, dW, db, dV, dc)

		return loss[-1], h[:,-1,:]


def sample(dataset, rnn, seed, n_sample=300):
	#h0, sample = None, None
	# inicijalizirati h0 na vektor nula
	# seed string pretvoriti u one-hot reprezentaciju ulaza

	h0_sample = np.zeros((1, rnn.hidden_size))
	
	seed = dataset.encode(seed)
	seed_oh = np.array([np.eye(rnn.vocab_size)[seed_] for seed_ in seed]).reshape(1,len(seed),rnn.vocab_size)

	h, _ = rnn.rnn_forward(seed_oh, h0_sample, rnn.U, rnn.W, rnn.b)

	o = []
	for t in range(rnn.sequence_length):
		o_t = softmax(rnn.output(h[:,t,:], rnn.V, rnn.c))
		o.append(np.argmax(o_t))
	sample = dataset.decode(o[:n_sample-len(seed)])

	return sample

def run_language_model(dataset, max_epochs, hidden_size=100, 								sequence_length=20, learning_rate=1e-2, 									sample_every=100):
	
	vocab_size = dataset.vocab_size
	rnn = RNN(vocab_size, sequence_length, hidden_size, learning_rate)

	print("U =", rnn.U.shape)
	print("W =", rnn.W.shape)
	print("b =", rnn.b.shape)
	print("V =", rnn.V.shape)
	print("c =", rnn.c.shape)

	current_epoch = 0 
	batch = 0

	h0 = np.zeros((batch_size, hidden_size))

	average_loss = 0
	print("Epoch {}/{}".format(current_epoch+1, max_epochs))
	while current_epoch < max_epochs: 
		e, x, y = dataset.next_minibatch()

		if e: 
			current_epoch += 1
			h0 = np.zeros((batch_size, hidden_size))

			average_loss /= dataset.num_batches
			print("Average loss =", average_loss)
			print("\nEpoch {}/{}".format(current_epoch+1, max_epochs))
			average_loss = 0
			# why do we reset the hidden state here?

		# One-hot transform the x and y batches
		x_oh = np.array([np.eye(rnn.vocab_size)[x_] for x_ in x])
		y_oh = np.array([np.eye(rnn.vocab_size)[y_] for y_ in y])

		# Run the recurrent network on the current batch
		# Since we are using windows of a short length of characters,
		# the step function should return the hidden state at the end
		# of the unroll. You should then use that hidden state as the
		# input for the next minibatch. In this way, we artificially
		# preserve context between batches.

		loss, h0 = rnn.step(h0, x_oh, y_oh)

		if batch % 100 == 0:
			print(loss)
			pass

		average_loss += loss
		if batch % sample_every == 0:
			seed ='HAN:\nIs that good or bad?\n\n'
			print(sample(dataset, rnn, seed))
		batch += 1


if __name__ == "__main__":

	dataset = Dataset('./data', batch_size=batch_size,
						sequence_length=seq_lenght)

	dataset.preprocess(input_file)
	dataset.create_minibatches()
	run_language_model(dataset, max_epochs=10, hidden_size=100)