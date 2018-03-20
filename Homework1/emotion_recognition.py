import tensorflow as tf
import os
import scipy.io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# a = tf.constant([1, 2], name='a')
# b = tf.constant([2, 3], name='b')
# result = a + b
# sess = tf.Session()
# print(sess.run(result))

INPUT_NODE = 310
OUTPUT_NODE = 1

LAYER1_NODE = 100

LEARNING_RATE = 0.8

TRAINING_STEPS = 2000


def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(x, weights1=weights1, biases1=biases1, weights2=weights2, biases2=biases2)
global_steps = tf.Variable(0, trainable=False)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE)\
    .minimize(cross_entropy_mean)


with tf.Session() as sess:
    x_data = sio.loadmat('./train_test/train_data.mat')
    x_data = x_data["train_data"]
    y_data = sio.loadmat('./train_test/train_label.mat')
    y_data = y_data["train_label"]
    tf.initialize_all_variables().run()
    for i in range(TRAINING_STEPS):
        sess.run(train_step, feed_dict={x: x_data, y_: y_data})
        if i % 50 == 0:
            print(sess.run(cross_entropy_mean, feed_dict={x: x_data, y_: y_data}))



