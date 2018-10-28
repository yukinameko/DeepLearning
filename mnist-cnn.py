import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
	initial = tf.truncated_normal(shape)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第1層 (入力層)
x = tf.placeholder("float", [None, 784])

# 形状変更
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第2層 (畳み込み層)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
y_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 第3層 (プーリング層)
y_pool1 = max_pool_2x2(y_conv1)

# 第4層 (畳み込み層)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
y_conv2 = tf.nn.relu(conv2d(y_pool1, W_conv2) + b_conv2)

# 第5層 (プーリング層)
y_pool2 = max_pool_2x2(y_conv2)

# 形状変更
y_pool2_flat = tf.reshape(y_pool2, [-1, 7 * 7 * 64])

# 第6層 (全結合層)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
y_fc1 = tf.nn.relu(tf.matmul(y_pool2_flat, W_fc1) + b_fc1)

# 第7層 (全結合層)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(y_fc1, W_fc2) + b_fc2)

# 損失関数を計算グラフを作成する
t = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(t * tf.log(y))

# 次の(1)、(2)を行うための計算グラフを作成する。
# (1) 損失関数に対するネットワークを構成するすべての変数の勾配を計算する。
# (2) 勾配方向に学習率分移動して、すべての変数を更新する。
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 初期化を行うための計算グラフを作成する。
init = tf.global_variables_initializer()

# テストデータに対する正答率を計算するための計算グラフを作成する。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# MNIST 入力データ
mnist = input_data.read_data_sets("data/", one_hot=True)

# セッションを作成して、計算グラフを実行する。
with tf.Session() as sess:
	# 初期化を実行する。
	sess.run(init)

	# 学習を実行する。
	for i in range(20000):
		x_batch, t_batch = mnist.train.next_batch(50)
		sess.run(train_step, feed_dict={x: x_batch, t: t_batch})

		if i % 100 == 0:
			result = sess.run(accuracy, feed_dict={x: mnist.test.images, t: mnist.test.labels})
			print(result)

	result = sess.run(accuracy, feed_dict={x: mnist.test.images, t: mnist.test.labels})
	print("accuracy:", result)