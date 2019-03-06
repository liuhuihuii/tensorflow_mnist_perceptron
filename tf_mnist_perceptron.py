# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:20:12 2019

@author: 86181
"""

##using tf implement one-layer perceptron

import tensorflow as tf

##load data##
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


##define##

x = tf.placeholder(tf.float32,[None,784])   ##input image

y_label = tf.placeholder(tf.float32,[None,10])

w = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y_predict = tf.nn.softmax(tf.matmul(x,w) + b)

loss = - tf.reduce_sum(y_label * tf.log(y_predict))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_label,1),tf.arg_max(y_predict,1)),tf.float32))
##run##
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(6000):
        xs, ys = mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:xs,y_label:ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y_label:mnist.test.labels})
        print('step:'+ str(i)+ ' accuracy is:' + str(acc))


print('Done!')
    

