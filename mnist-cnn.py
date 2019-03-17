import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('data/', one_hot=True)

x = tf.placeholder(tf.float32, name='x')

x1 = tf.reshape(x, [-1, 28, 28, 1])

k0 = tf.Variable(tf.truncated_normal([5, 5, 1, 48], mean=0.0, stddev=0.1))
x2 = tf.nn.relu(tf.nn.conv2d(x1, k0, strides=[1, 1, 1, 1], padding='SAME'))
x3 = tf.nn.max_pool(x2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

k1 = tf.Variable(tf.truncated_normal([5, 5, 48, 96]))
x4 = tf.nn.relu(tf.nn.conv2d(x3, k1, strides=[1,1,1,1], padding='SAME'))
x4_ = tf.nn.max_pool(x4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

x5 = tf.reshape(x4_, [-1, 7*7*96])

w6 = tf.Variable(tf.zeros([7*7*96, 512]))
b6 = tf.Variable([0.1] * 512)
x6 = tf.matmul(x5, w6) + b6

x7 = tf.nn.relu(x6)

w8 = tf.Variable(tf.zeros([512, 10]))
b8 = tf.Variable([0.1] * 10)
x8 = tf.matmul(x7, w8) + b8

y = tf.nn.softmax(x8)

labels = tf.placeholder(tf.float32, name='labels')
loss = -tf.reduce_sum(labels * tf.log(y))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

prediction_match = tf.equal(tf.argmax(y, axis=1), tf.argmax(labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction_match, tf.float32), name='accuracy')

BATCH_SIZE = 32
NUM_TRAIN = 10_000
OUTPUT_BY = 500

sess.run(tf.global_variables_initializer())
for i in range(NUM_TRAIN):
	x_batch, labels_batch = mnist.train.next_batch(BATCH_SIZE)
	inout = {x: x_batch, labels: labels_batch}
	if i % OUTPUT_BY == 0:
		train_accuracy = accuracy.eval(feed_dict=inout)
		print('step {:d}, accuracy {:.2f}'.format(i, train_accuracy))

	optimizer.run(feed_dict=inout)

test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, labels: mnist.test.labels})
print('test accuracy {:.2f}'.format(test_accuracy))
