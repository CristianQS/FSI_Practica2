# -*- coding: utf-8 -*-
import gzip
import pickle
import matplotlib.pyplot as plt
from matplotlib.dates import epoch2num

import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)#,encoding='latin1')
f.close()

# TODO: the neural net!!

x_training, y_training = train_set
x_validation, y_validation = valid_set
x_test, y_test = test_set

y_training = one_hot(y_training,10)
y_validation = one_hot(y_validation,10)
y_test = one_hot(y_test,10)


x = tf.placeholder(tf.float32, [None, 784])  # samples
y_ = tf.placeholder(tf.float32, [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)


h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
# Train
noError = True
epoch = 0
error_list = []
epoch_list = []
while noError and epoch < 500:
    for jj in xrange(len(x_training) / batch_size):
        batch_xs = x_training[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_training[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_validation, y_: y_validation})
    print "Epoch #:", epoch, "Error: ",error
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r

    if error < 0.1:
        noError = False

    error_list.append(error)
    epoch_list.append(epoch)
    epoch += 1
    print "----------------------------------------------------------------------------------"
print "----------------------"
print "        TEST  "
print "----------------------"
plt.show()

test = sess.run(loss, feed_dict={x: x_test, y_: y_test})
print "Error del test ha sido de: ", test


equalResults = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(equalResults, tf.float32))
porcentaje = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})*100
print "El porcentaje de acierto es de: %f" % porcentaje



