#!/usr/bin/python
import tensorflow as tf
import numpy as np
X = np.array([[0,0],[1,0],[0,1],[1,1]])
Y = np.array([[0],[1],[1],[0]])
n1 = 10
n2 = 10
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w1 = tf.Variable(tf.random_uniform([2,10],-1,1))
w2 = tf.Variable(tf.random_uniform([10,10],-1,1))
w3 = tf.Variable(tf.random_uniform([10,1],-1,1))

b1 = tf.Variable(tf.zeros([10]))+1
b2 = tf.Variable(tf.zeros([10]))+1
b3 = tf.Variable(tf.zeros([1]))+1

l1 = tf.sigmoid(tf.matmul(x,w1)+b1)
l2 = tf.sigmoid(tf.matmul(l1,w2)+b2)
lo = tf.sigmoid(tf.matmul(l2,w3)+b3)
learning_rate = 1
epochs = 100000
cost = tf.reduce_mean(-y*tf.log(lo)-(1-y)*tf.log(1-lo))
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in xrange(epochs):
        sess.run(opt, feed_dict ={x:X,y:Y})
        if i % 1000 ==0:
            print sess.run(cost, feed_dict={x:X,y:Y})
    ans = tf.equal(tf.floor(lo+0.5),y)
    accuracy = tf.reduce_mean(tf.cast(ans,"float"))

    print sess.run([lo],feed_dict={x:X,y:Y})
    print "accuracy: ", accuracy.eval({x:X,y:Y})*100," %"
