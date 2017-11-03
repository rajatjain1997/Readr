import tensorflow as tf

n_nodes_hl1 = 50

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 100])
y = tf.placeholder('float')
beta = tf.Variable(1.0)

def log_sigmoid(x,beta):
	return tf.div(1,1+tf.exp(tf.mul(-beta,x)))

def neural_network_model(data,beta):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([100, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = log_sigmoid(l1,beta)

    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']

    return output

# def train_neural_network(x,beta):
#     prediction = neural_network_model(x,beta)
#     cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
#     optimizer = tf.train.AdamOptimizer().minimize(cost)
    
#     hm_epochs = 10
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             for _ in range(int(mnist.train.num_examples/batch_size)):
#                 epoch_x, epoch_y = mnist.train.next_batch(batch_size)
#                 _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
#                 epoch_loss += c

#             print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#         print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)