import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

mnist = input_data.read_data_sets("data/", one_hot=True)
layers = [784, 30, 10]
batchSize = 1000
maxEpochs = 10

x = tf.placeholder('float', [None, layers[0]])
y = tf.placeholder('float', [None, layers[-1]])
beta = tf.Variable(2.0)
learningRate = tf.constant(1.0)

layerParameters = [{
	'weights':tf.Variable(tf.random_normal([layers[i-1], layers[i]])),
	'biases':tf.Variable(tf.random_normal([layers[i]]))
} for i in range(1, len(layers))]


def log_sigmoid(x):
	return tf.divide(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.multiply(-beta,x))))

def diff_log_sigmoid(x):
	k = log_sigmoid(x)
	return tf.multiply(beta, tf.multiply(k, tf.subtract(tf.constant(1.0), k)))

def neural_network(x):
	outputList = [x]
	netList = []
	for i in range(len(layers)-1):
		layerNet = tf.add(tf.matmul(outputList[i], layerParameters[i]['weights']), layerParameters[i]['biases'])
		netList.append(layerNet)
		outputList.append(log_sigmoid(layerNet))

	return outputList, netList

def train_neural_network(x):
	outputList, netList = neural_network(x)
	prediction = outputList[-1]
	biasOP = tf.transpose(tf.constant([[1.0 for i in range(batchSize)]]))

	# print(prediction, netList, outputList)

	S = [tf.multiply(tf.constant(-2.0), tf.multiply(diff_log_sigmoid(netList[-1]), tf.subtract(y, prediction)))]
	for i in range(len(layers)-2, 0, -1):
		Slayer = tf.multiply(diff_log_sigmoid(netList[i-1]), tf.transpose(tf.matmul(layerParameters[i]["weights"], tf.transpose(S[0]))))
		S.insert(0, Slayer)

	results = [
		[
		tf.assign(layerParameters[i]['weights'],tf.subtract(layerParameters[i]['weights'], tf.multiply(learningRate, tf.transpose(tf.matmul(tf.transpose(S[i]), outputList[i]))))),
		tf.assign(layerParameters[i]['biases'],tf.subtract(layerParameters[i]['biases'], tf.multiply(learningRate, tf.transpose(tf.matmul(tf.transpose(S[i]), biasOP)))[0]))
		] for i in range(len(S))
	]


	acct_mat = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# sess = tf_debug.LocalCLIDebugWrapperSession(sess)

		for epoch in range(maxEpochs):
			for i in range(int(mnist.train.num_examples/batchSize)):
				epoch_x, epoch_y = mnist.train.next_batch(batchSize)
				l=sess.run([results, outputList], feed_dict={x: epoch_x, y: epoch_y})
				print(l[1][2])
			res = sess.run(acct_res, feed_dict =
				{
					x: mnist.test.images[:1000],
					y : mnist.test.labels[:1000]
				}
			)
			print(res)

train_neural_network(x)