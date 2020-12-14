import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from data import *

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

N=mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]

"""
image = mnist.train.images[1,:].reshape(28,28)
plt.imshow(image, cmap = plt.get_cmap('Blues'), vmin = 0, vmax = 1)
plt.show()
"""

# Train/Test Data
X_train = mnist.train.images
Y_train = mnist.train.labels

X_test = mnist.test.images
Y_test = mnist.test.labels


# Defining inputs and outputs
X = tf.placeholder(tf.float32, [None, 28*28]) # image dimensions = 28*28
Yoh_ = tf.placeholder(tf.float32, [None, 10]) # 10 classes for 10 digits

# Creating [784, 10] neural net
W = tf.Variable(tf.random_normal([D, C]), name="W")
b = tf.Variable(tf.zeros([C]), name="b")


# Output probabilites
logits = tf.matmul(X, W) + b
probs = tf.nn.softmax(logits)

# Defining loss function
cross_entropy = tf.losses.softmax_cross_entropy(Yoh_, logits)
cost = tf.reduce_mean(cross_entropy)
cost_history = []

# Optimizer
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer_2 = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(cost)

# Initializing variables
sess = tf.Session()
init = tf.initializers.global_variables()
sess.run(init)

# Learning
epochs = 1000
i = 0
for i in range(epochs):

	feed_dict={X:X_train, Yoh_:Y_train}
	cost_val = sess.run(cost, feed_dict=feed_dict)
	cost_history.append(cost_val)

	sess.run(train_op, feed_dict=feed_dict)

	if i%10 == 0:
		print("iteration {}: cost {}".format(i, cost_val))

# Testing network
probs_val = sess.run(probs, feed_dict={X:X_test})
Y_predicted = np.argmax(probs_val, axis=1)

# Performance
accuracy, pr, M = eval_perf_multi(Y_predicted, np.argmax(Y_test, axis=1))
print("Accuracy: ", accuracy)
print("Precision / Recall")
for i in range(len(pr)):
	print("Class {} : {}".format(i, pr[i]))
print("Confusion Matrix:\n ", M)

# Ploting cost history
plt.plot(range(len(cost_history)), cost_history)
plt.ylabel("cost")
plt.xlabel("epochs")
plt.show()