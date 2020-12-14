import tensorflow as tf
import os
import pickle
import numpy as np
import time
import math
import skimage as ski
import skimage.io
from sklearn import preprocessing
import matplotlib.pyplot as plt

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10
num_epochs = 8
batch_size = 100

DATA_DIR = './data/CIFAR-10'
FILTERS_SAVE_DIR = './output/zad4/filters'
PLOTS_SAVE_DIR = './output/zad4/plots'

lr_policy = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

LOSS_SCALE = 1e-3
def build_cnn(X, Y_, lr):

    # Input Layer
    input_layer = tf.reshape(X, [-1, img_width, img_height, num_channels])

    # regularisers
    l2_reg1 = tf.contrib.layers.l2_regularizer(
                                scale = LOSS_SCALE)

    l2_reg2 = tf.contrib.layers.l2_regularizer(
                                scale = LOSS_SCALE)

    l2_reg3 = tf.contrib.layers.l2_regularizer(
                                scale = LOSS_SCALE)

    l2_reg4 = tf.contrib.layers.l2_regularizer(
                                scale = LOSS_SCALE)

    # convolutional layer 1
    conv1 = tf.layers.conv2d(
                inputs = input_layer,
                filters = 16,
                kernel_size = [5, 5],
                padding = "same",
                activation = tf.nn.relu,
                kernel_regularizer = l2_reg1,
                name = 'conv1')

    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(
                inputs = conv1,
                pool_size = 3,
                strides = 2)

    # convolutional layer 2
    conv2 = tf.layers.conv2d(
                inputs = pool1,
                filters = 32,
                kernel_size = [5, 5],
                padding = "same",
                activation = tf.nn.relu,
                kernel_regularizer = l2_reg2)

    # pooling layer 2
    pool2 = tf.layers.max_pooling2d(
                inputs = conv2,
                pool_size = 3,
                strides = 2)

    # flatten
    flat = tf.layers.flatten(inputs = pool1)
    
    # fully connected layer 1
    fc1 = tf.layers.dense(
                inputs = flat,
                units = 128,
                activation = tf.nn.relu,
                kernel_regularizer = l2_reg3)
    
    # fully connected layer 2
    fc2 = tf.layers.dense(
                inputs = flat,
                units = 128,
                activation = tf.nn.relu,
                kernel_regularizer = l2_reg4)

    # logits layer
    logits = tf.layers.dense(
                inputs = fc2,
                units = 10)

    # loss calculation
    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.losses.sparse_softmax_cross_entropy(labels=Y_, logits=logits)
    total_loss = loss + l2_loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(total_loss)

    return logits, total_loss, train_op, conv1


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
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)


def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.pdf')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y


def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


def evaluate(name, sess, logits, loss, X, Y_, x, y):

  print("\nRunning evaluation: %s..." %(name))
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size

  cnt_correct = 0
  loss_avg = 0
  for i in range(num_batches):
    offset = i * batch_size 
    batch_x = x[offset:(offset + batch_size), ...]
    batch_y = y[offset:(offset + batch_size)]

    logits_val = sess.run(logits, feed_dict={X:batch_x})

    yp = np.argmax(logits_val, 1)
    cnt_correct += (yp == batch_y).sum()

    loss_val = sess.run(loss, feed_dict={Y_:batch_y, logits:logits_val})
    loss_avg += loss_val

  acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches

  print(name + " accuracy = %.2f" % acc)
  print(name + " avg loss = %.2f\n" % loss_avg)
  return loss_avg, acc


def main():

  train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
  train_y = []

  for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
  train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
  train_y = np.array(train_y, dtype=np.int32)

  subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
  test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
  test_y = np.array(subset['labels'], dtype=np.int32)

  valid_size = 5000
  train_x, train_y = shuffle_data(train_x, train_y)
  valid_x = train_x[:valid_size, ...]
  valid_y = train_y[:valid_size, ...]
  train_x = train_x[valid_size:, ...]
  train_y = train_y[valid_size:, ...]
  data_mean = train_x.mean((0,1,2))
  data_std = train_x.std((0,1,2))

  train_x = (train_x - data_mean) / data_std
  valid_x = (valid_x - data_mean) / data_std
  test_x = (test_x - data_mean) / data_std

  plot_data = {}
  plot_data['train_loss'] = []
  plot_data['valid_loss'] = []
  plot_data['train_acc'] = []
  plot_data['valid_acc'] = []
  plot_data['lr'] = []
  
  X = tf.placeholder(dtype=tf.float32, shape=[None, img_width, img_height,num_channels])
  Y_ = tf.placeholder(dtype=tf.int32, shape=[None])
  lr = tf.placeholder(tf.float32)

  logits, total_loss, train_op, conv1 = build_cnn(X,Y_, lr)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Training model

  for epoch_num in range(1, num_epochs + 1):
    train_x, train_y = shuffle_data(train_x, train_y)

    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    
    if epoch_num in lr_policy:
      solver_config = lr_policy[epoch_num]

    for step in range(num_batches):
      offset = step * batch_size 
      # s ovim kodom pazite da je broj primjera djeljiv s batch_size
      batch_x = train_x[offset:(offset + batch_size), ...]
      batch_y = train_y[offset:(offset + batch_size)]

      feed_dict = {X: batch_x, Y_: batch_y, lr:solver_config['lr']}
      run_ops = [train_op, total_loss, logits]

      start_time = time.time()

      ret_val = sess.run(run_ops, feed_dict=feed_dict)
      _, loss_val, logits_val = ret_val

      duration = time.time() - start_time

      if (step+1) % 50 == 0:
        sec_per_batch = float(duration)
        format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
        print(format_str % (epoch_num, step+1, num_batches, loss_val, sec_per_batch))
      	
      if step % 100 == 0:  
        conv1_var = tf.contrib.framework.get_variables('conv1')[0]
        conv1_weights = conv1_var.eval(session=sess)
        draw_conv_filters(epoch_num, step, conv1_weights, FILTERS_SAVE_DIR)

    train_loss, train_acc = evaluate("Train", sess, logits, total_loss, X, Y_, train_x, train_y)
    valid_loss, valid_acc = evaluate("Validation", sess, logits, total_loss, X, Y_, valid_x, valid_y)

    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [valid_loss]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [valid_acc]
    plot_data['lr'] += [lr.eval(session=sess, 
                        feed_dict={lr:solver_config['lr']})]
    plot_training_progress(PLOTS_SAVE_DIR, plot_data)

if __name__ == "__main__":
    main()