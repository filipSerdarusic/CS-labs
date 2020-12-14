import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import math
import os
import skimage as ski
import skimage.io

DATA_DIR = './data/MNIST'
SAVE_DIR = "./output/zad3"

config = {}
config['max_epochs'] = 8
config['batch_size'] = 250
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}


def build_cnn(X, Y_, num_filters=16, kernel_size=[5,5], padding="same"):

	# Input layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(X, [-1, 28, 28, 1])

	# Add regularization
	# ...

	# convolutional layer 1
	conv1 = tf.layers.conv2d(
				inputs = input_layer,
				filters = num_filters,
				kernel_size = kernel_size,
				padding = padding,
				activation = tf.nn.relu,
				name = 'conv1')

	# pooling layer 1
	pool1 = tf.layers.max_pooling2d(
				inputs = conv1,
				pool_size = [2,2],
				strides = 2)
	
	# convolutional layer 2
	conv2 = tf.layers.conv2d(
				inputs = pool1,
				filters = 2*num_filters,
				kernel_size = kernel_size,
				padding = padding,
				activation = tf.nn.relu,
				name = 'conv2')

	# pooling layer 2
	pool2 = tf.layers.max_pooling2d(
				inputs = conv2,
				pool_size = [2,2],
				strides = 2)
	
	# flatten layer
	flat = tf.layers.flatten(inputs = pool2)

	# fully connected layer
	fc = tf.layers.dense(
			inputs = flat,
			units = 512,
			activation = tf.nn.relu)
	
	# logits layer
	logits = tf.layers.dense(
				inputs = fc,
				units = 10)

	# calculating loss
	# Must add regularization loss!
	loss = tf.losses.softmax_cross_entropy(onehot_labels=Y_, logits=logits)

	return logits, loss, conv1


def draw_conv_filters(epoch, step, weights, save_dir):
	w = weights.copy()
	num_filters = w.shape[3]
	num_channels = w.shape[2]
	k = w.shape[0]
	assert w.shape[0] == w.shape[1]
	w = w.reshape(k, k, num_channels, num_filters)
	w -= w.min()
	w /= w.max()
	border = 1
	cols = 8
	rows = math.ceil(num_filters / cols)
	width = cols * k + (cols-1) * border
	height = rows * k + (rows-1) * border
	img = np.zeros([height, width, 3])
	for i in range(num_filters):
		r = int(i / cols) * (k + border)
		c = int(i % cols) * (k + border)
		img[r:r+k,c:c+k,:] = w[:,:,:,i]
	filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
	ski.io.imsave(os.path.join(save_dir, filename), img)


def train(sess, train_x, train_y, valid_x, valid_y, X, Y_, lr,
		logits, loss, optimizer, config):

	lr_policy = config['lr_policy']
	batch_size = config['batch_size']
	max_epochs = config['max_epochs']
	save_dir = config['save_dir']

	num_examples = train_x.shape[0]
	assert num_examples % batch_size == 0
	num_batches = num_examples // batch_size

	for epoch in range(1, max_epochs+1):

		if epoch in lr_policy:
			solver_config = lr_policy[epoch]

		cnt_correct = 0
		#for i in range(num_batches):
		# shuffle the data at the beggining of each epoch
		permutation_idx = np.random.permutation(num_examples)
		train_x = train_x[permutation_idx]
		train_y = train_y[permutation_idx]
		#for i in range(100):

		for i in range(num_batches):

			# store mini-batch to ndarray
			batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
			batch_y = train_y[i*batch_size:(i+1)*batch_size, :]

			logits_val, loss_val, _ = sess.run([logits, loss, optimizer], feed_dict={X:batch_x, Y_:batch_y, lr:solver_config['lr']})

			# compute classification accuracy
			yp = np.argmax(logits_val, 1)
			yt = np.argmax(batch_y, 1)
			cnt_correct += (yp == yt).sum()

			if i % 5 == 0:
				print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))

			if i % 100 == 0:
				conv1_var = tf.contrib.framework.get_variables('conv1')[0]
				conv1_weights = conv1_var.eval(session=sess)
				draw_conv_filters(epoch, i, conv1_weights, save_dir)

			if i > 0 and i % 50 == 0:
				print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))

		print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
		evaluate("Validation", sess, valid_x, valid_y, X, Y_, logits, loss, config)


def evaluate(name, sess, x, y, X, Y_, logits, loss, config):
	print("\nRunning evaluation: ", name)
	batch_size = config['batch_size']
	num_examples = x.shape[0]
	assert num_examples % batch_size == 0
	num_batches = num_examples // batch_size
	cnt_correct = 0
	loss_avg = 0
	for i in range(num_batches):
		batch_x = x[i*batch_size:(i+1)*batch_size, :]
		batch_y = y[i*batch_size:(i+1)*batch_size, :]

		logits_val = sess.run(logits, feed_dict={X:batch_x})

		yp = np.argmax(logits_val, 1)
		yt = np.argmax(batch_y, 1)
		cnt_correct += (yp == yt).sum()

		loss_val = sess.run(loss, feed_dict={Y_:batch_y, logits:logits_val})
		loss_avg += loss_val
		#print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
	valid_acc = cnt_correct / num_examples * 100
	loss_avg /= num_batches
	print(name + " accuracy = %.2f" % valid_acc)
	print(name + " avg loss = %.2f\n" % loss_avg)


def main():
	# Load data
	np.random.seed(int(time.time() * 1e6) % 2 ** 31)
	dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
	train_x = dataset.train.images
	train_x = train_x.reshape([-1, 28, 28, 1])
	train_y = dataset.train.labels
	valid_x = dataset.validation.images
	valid_x = valid_x.reshape([-1, 28, 28, 1])
	valid_y = dataset.validation.labels
	test_x = dataset.test.images
	test_x = test_x.reshape([-1, 28, 28, 1])
	test_y = dataset.test.labels
	train_mean = train_x.mean()
	train_x -= train_mean
	valid_x -= train_mean
	test_x -= train_mean

	X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
	Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

	logits, loss, conv1 = build_cnn(X,Y_)

	lr = tf.placeholder(tf.float32)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)							.minimize(loss)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	train(sess, train_x, train_y, valid_x, valid_y, X, Y_, lr,
			logits, loss, optimizer, config)

if __name__ == "__main__":
	main()