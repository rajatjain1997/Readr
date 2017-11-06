#!/usr/bin/env python
from image_conversion import convert
from gainfuzzify import gain
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
learningRate = tf.constant(0.5)
beta = tf.Variable(1.0)

middle = 30
# weight1 = tf.Variable(tf.random_normal([784, middle]))
# bias1 = tf.Variable(tf.random_normal([1, middle]))
# weight2 = tf.Variable(tf.random_normal([middle, 10]))
# bias2 = tf.Variable(tf.random_normal([1, 10]))

weight1 = tf.Variable(tf.truncated_normal([784, middle]))
bias1 = tf.Variable(tf.truncated_normal([1, middle]))
weight2 = tf.Variable(tf.truncated_normal([middle, 10]))
bias2 = tf.Variable(tf.truncated_normal([1, 10]))


def logSigmoid(x):
	return tf.divide(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(tf.multiply(beta,x)))))

def diffLogSigmoid(x):
	return tf.multiply(beta, tf.multiply(logSigmoid(x), tf.subtract(tf.constant(1.0), logSigmoid(x))))


net1 = tf.add(tf.matmul(x, weight1), bias1)
out1 = logSigmoid(net1)
net2 = tf.add(tf.matmul(out1, weight2), bias2)
out2 = logSigmoid(net2)

diff = tf.subtract(out2, y)

S2 = tf.multiply(tf.constant(-2.0),tf.multiply(diff, diffLogSigmoid(net2)))
deltaBias2 = S2
deltaWeight2 = tf.matmul(tf.transpose(out1), S2)
 
S1 = tf.multiply(tf.matmul(S2, tf.transpose(weight2)), diffLogSigmoid(net1))
deltaBias1 = S1
deltaWeight1 = tf.matmul(tf.transpose(x), S1)

s1 = tf.reduce_mean(S1)
s2 = tf.reduce_mean(S2)
s1Abs = tf.reduce_mean(tf.abs(S1))
s2Abs = tf.reduce_mean(tf.abs(S2))
s1 = tf.cond((tf.less(s1,0)), lambda: tf.negative(s1Abs), lambda: s1Abs)
s2 = tf.cond((tf.less(s2,0)), lambda: tf.negative(s2Abs), lambda: s2Abs)

result = [
	tf.assign(weight1,
			tf.add(weight1, tf.multiply(learningRate, deltaWeight1)))
  , tf.assign(bias1,
			tf.add(bias1, tf.multiply(learningRate,
							   tf.reduce_mean(deltaBias1, axis=[0]))))
  , tf.assign(weight2,
			tf.add(weight2, tf.multiply(learningRate, deltaWeight2)))
  , tf.assign(bias2,
			tf.add(bias2, tf.multiply(learningRate,
							   tf.reduce_mean(deltaBias2, axis=[0]))))
]

acct_mat = tf.equal(tf.argmax(out2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

def checkAccuracy(sess, numTestingSamples):
	return sess.run(acct_res, feed_dict =
						   {x: mnist.test.images[:numTestingSamples],
							y : mnist.test.labels[:numTestingSamples]})


def provideMnistTraining(sess, numTrainingSamples):

	for i in range(numTrainingSamples):
		batch_xs, batch_ys = mnist.train.next_batch(1)
		l = sess.run([result,s1,s2], feed_dict = {x: batch_xs,
									y : batch_ys})
		# print(l[1], " ", l[2])
		l = sess.run(tf.assign(beta, tf.constant(gain(l[1], l[2])/2, dtype=tf.float32)))
		# print("beta:", l)
		if i % 100 == 0:
			res = checkAccuracy(sess, 1000)
			print(i/100 + 1, ": ", res)
	print("MNIST training complete")

def read(sess, imagepath):
	# Image conversion goes here
	image = convert(imagepath)
	# Stil need to test it Coz we sort of don't know the representation of the mnist dataset
	# leo says that 0 is considered white and 255 is considered black. Chutiya.
	return sess.run(out2, feed_dict = {x: image})

def outputCharacter(sess, imagepath):
	output = read(sess, imagepath)
	index = np.argmax(output)
	if (index < 10):
		return index
	else :
		return chr(index+55)


def reset(sess):
	sess.run([
		tf.assign(weight1, tf.truncated_normal([784, middle])),
		tf.assign(bias1, tf.truncated_normal([1, middle])),
		tf.assign(weight2, tf.truncated_normal([middle, 10])),
		tf.assign(bias2, tf.truncated_normal([1, 10]))
		])
	print("Network Reset!")


def restoreModel(session,fileName):
    saver = tf.train.Saver()
    saver.restore(session,fileName)
    return session

def session():
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	return sess

def storeModel(session,fileName):
    saver = tf.train.Saver()
    saver.save(session,fileName)

def train(sess, imagepath, actualresult):
	image = convert(imagepath)
	l = sess.run([result, s1, s2], feed_dict = {x: image,
									y:actualresult})
	sess.run(tf.assign(beta, tf.constant(gain(l[1], l[2])/2, dtype=tf.float32)))
	print("Trained Model with the new image!")

provideMnistTraining(session(), 10000, 'MnistTrainedModel')
