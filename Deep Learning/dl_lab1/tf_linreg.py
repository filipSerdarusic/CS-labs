import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

## 1. Defining the graph
X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])

a = tf.Variable(0.0)
b = tf.Variable(0.0)

# Linear regression model
Y = a * X + b

# Square loss
loss = tf.reduce_mean((Y-Y_)**2)
loss_history = []

# Gradient Descent optimization
trainer = tf.train.GradientDescentOptimizer(learning_rate=10e-4)
train_op = trainer.minimize(loss)

# Manually computing and applying gradients
grads_and_vars = trainer.compute_gradients(loss, [a, b])
train_op_2 = trainer.apply_gradients(grads_and_vars)

## 2. Initializing variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Creating data 
x = np.arange(50)
y = 2*x + 1.0
y += np.random.normal(scale=3, size=len(x))

plt.scatter(x,y)

## 3. Learning
for i in range(int(1e4)):

  val_loss, _, val_a,val_b = sess.run([loss, train_op, a, b],
                               feed_dict={X: x, Y_: y})

  loss_history.append(val_loss)
  
  if i%10 == 0:
    print(i,val_loss, val_a,val_b)

print("Optimization finished!")

print("a =", val_a, "\nb =", val_b)

plt.plot(x, x*val_a + val_b, color='red')
#plt.grid()
plt.show()