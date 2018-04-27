import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.001
TRAIN_STEP = 10000
TRAIN_SET_NUM = 9
TEST_SET_SUM = 6
BATCH_SIZE = 100
INPUT_SIZE = 310
TIME_STEP = 10
HIDDEN_NODE_NUM = 256
LAYER_NUM = 2
OUTPUT_SIZE = 3

x_ = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_NODE_NUM])),
    'out': tf.Variable(tf.random_normal([HIDDEN_NODE_NUM, OUTPUT_SIZE]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_NODE_NUM, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE, ]))
}


def rnn(X, weights, biases):
    X = tf.reshape(X, [-1, INPUT_SIZE])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, TIME_STEP, INPUT_SIZE])
    lstm = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_NODE_NUM, forget_bias=1.0)
    init_state = lstm.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm, x_, initial_state=init_state, time_major=False)

    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


pred = rnn(x_, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_train_test_data(number):
    data = np.load('./data_used_10399/0' + number + '.npz')
    files_in_zip = data.keys()
    train_data = []
    test_data = []
    for i in range(0, TRAIN_SET_NUM):
        train_data.append([])
        clip = data[files_in_zip[i]]
        shape = clip.shape
        for j in range(0, shape[1]):
            train_data[i].append([])
            for m in range(0, shape[0]):
                for n in range(0, shape[2]):
                    train_data[i][j].append(clip[m][j][n])

    for i in range(0, TEST_SET_SUM):
        test_data.append([])
        clip = data[files_in_zip[TRAIN_SET_NUM + i]]
        shape = clip.shape
        for j in range(0, shape[1]):
            test_data[i].append([])
            for m in range(0, shape[0]):
                for n in range(0, shape[2]):
                    test_data[i][j].append(clip[m][j][n])
        i += 1
    return train_data, test_data


def get_train_test_label():
    label = np.load('./data_used_10399/label.npy')
    train_label = label[0:TRAIN_SET_NUM]
    test_label = label[TRAIN_SET_NUM:TRAIN_SET_NUM + TEST_SET_SUM]
    train_one_hot = []
    test_one_hot = []
    for i in range(0, train_label.size):
        train_one_hot.append([0, 0, 0])
        train_one_hot[i][train_label[i]] = 1
    for i in range(0, test_label.size):
        test_one_hot.append([0, 0, 0])
        test_one_hot[i][test_label[i]] = 1
    return train_label, test_label


init = tf.global_variables_initializer()
train_data_1, test_data_1 = get_train_test_data('1')
train_data_2, test_data_2 = get_train_test_data('2')
train_data_3, test_data_3 = get_train_test_data('3')
train_label, test_label = get_train_test_label()
#
# with tf.Session() as sess:
#     sess.run(init)
#     step = 0
#     while step * BATCH_SIZE < training_iters:
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
#         sess.run([train_op], feed_dict={
#             x: batch_xs,
#             y: batch_ys,
#         })
#         if step % 20 == 0:
#             print(sess.run(accuracy, feed_dict={
#             x: batch_xs,
#             y: batch_ys,
#         }))
#         step += 1


