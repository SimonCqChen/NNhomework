import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.01
TRAIN_STEP = 1000
TRAIN_SET_NUM = 9
TEST_SET_SUM = 6
INPUT_SIZE = 310
# 15段视频，最短的是185s
SHORTEST_LENGTH = 185
TIME_STEP = 5
HIDDEN_NODE_NUM = 256
OUTPUT_SIZE = 3
BATCH_SIZE = 9 * int(SHORTEST_LENGTH / TIME_STEP)

x_ = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
batch_size = tf.placeholder(tf.int32, [])

weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_NODE_NUM])),
    'out': tf.Variable(tf.random_normal([HIDDEN_NODE_NUM, OUTPUT_SIZE]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_NODE_NUM, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE, ]))
}


def rnn(X, weights, biases, batch_size):
    X = tf.reshape(X, [-1, INPUT_SIZE])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, TIME_STEP, HIDDEN_NODE_NUM])
    lstm = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_NODE_NUM, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results


pred = rnn(x_, weights, biases, batch_size)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_train_test_data(number):
    data = np.load('./data_used_10399/0' + number + '.npz')
    files_in_zip = data.keys()
    train_data = []
    test_data = []
    for i in range(0, TRAIN_SET_NUM):
        train_data.append([])
        clip = data[files_in_zip[i]]
        shape = clip.shape
        for j in range(0, SHORTEST_LENGTH):
            train_data[i].append([])
            for m in range(0, shape[0]):
                for n in range(0, shape[2]):
                    train_data[i][j].append(clip[m][j][n])

    for i in range(0, TEST_SET_SUM):
        test_data.append([])
        clip = data[files_in_zip[TRAIN_SET_NUM + i]]
        shape = clip.shape
        for j in range(0, SHORTEST_LENGTH):
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
    return train_one_hot, test_one_hot


train_data_1, test_data_1 = get_train_test_data('1')
train_data_2, test_data_2 = get_train_test_data('2')
train_data_3, test_data_3 = get_train_test_data('3')
train_label_temp, test_label_temp = get_train_test_label()
train_data_1 = np.reshape(train_data_1, (-1, TIME_STEP, INPUT_SIZE))
train_data_2 = np.reshape(train_data_2, (-1, TIME_STEP, INPUT_SIZE))
train_data_3 = np.reshape(train_data_3, (-1, TIME_STEP, INPUT_SIZE))
test_data_1 = np.reshape(test_data_1, (-1, TIME_STEP, INPUT_SIZE))
test_data_2 = np.reshape(test_data_2, (-1, TIME_STEP, INPUT_SIZE))
test_data_3 = np.reshape(test_data_3, (-1, TIME_STEP, INPUT_SIZE))
train_label = []
test_label = []
for label in train_label_temp:
    for i in range(0, int(SHORTEST_LENGTH / TIME_STEP)):
        train_label.append(label)
for label in test_label_temp:
    for j in range(0, int(SHORTEST_LENGTH / TIME_STEP)):
        test_label.append(label)

init = tf.global_variables_initializer()


def run_lstm():
    with tf.Session() as sess:
        accuracy_file = open('accuracy_file_01.csv', 'w')
        accuracy_file.write('step,accuracy\n')
        cost_file = open('cost_file_01.csv', 'w')
        cost_file.write('step,cost\n')
        sess.run(init)
        for step in range(0, TRAIN_STEP):
            x, y = train_data_1, train_label
            sess.run([train_op], feed_dict={
                x_: x,
                y_: y,
                batch_size: 333
            })
            if step % 50 == 0:
                print(
                    sess.run(correct_prediction, feed_dict={
                        x_: test_data_1,
                        y_: test_label,
                        batch_size: 222})
                    # sess.run(weights['out'])
                    )
                print('======================================')
                cost_value = sess.run(cost, feed_dict={
                        x_: x,
                        y_: y,
                        batch_size: 333})
                accu_value = sess.run(accuracy, feed_dict={
                    x_: test_data_1,
                    y_: test_label,
                    batch_size: 222})
                accuracy_file.write(str(step) + ',' + str(accu_value) + '\n')
                cost_file.write(str(step) + ',' + str(cost_value) + '\n')
        accuracy_file.close()
        cost_file.close()

    with tf.Session() as sess:
        accuracy_file = open('accuracy_file_02.csv', 'w')
        accuracy_file.write('step,accuracy\n')
        cost_file = open('cost_file_02.csv', 'w')
        cost_file.write('step,cost\n')
        sess.run(init)
        for step in range(0, TRAIN_STEP):
            x, y = train_data_2, train_label
            sess.run([train_op], feed_dict={
                x_: x,
                y_: y,
                batch_size: 333
            })
            if step % 50 == 0:
                # print(
                #     sess.run(cost, feed_dict={
                #         x_: test_data_1,
                #         y_: test_label,
                #         batch_size: 222})
                #     # sess.run(weights['out'])
                #     )
                # print('======================================')
                cost_value = sess.run(cost, feed_dict={
                        x_: x,
                        y_: y,
                        batch_size: 333})
                accu_value = sess.run(accuracy, feed_dict={
                    x_: test_data_2,
                    y_: test_label,
                    batch_size: 222})
                accuracy_file.write(str(step) + ',' + str(accu_value) + '\n')
                cost_file.write(str(step) + ',' + str(cost_value) + '\n')
        accuracy_file.close()
        cost_file.close()

    with tf.Session() as sess:
        accuracy_file = open('accuracy_file_03.csv', 'w')
        accuracy_file.write('step,accuracy\n')
        cost_file = open('cost_file_03.csv', 'w')
        cost_file.write('step,cost\n')
        sess.run(init)
        for step in range(0, TRAIN_STEP):
            x, y = train_data_3, train_label
            sess.run([train_op], feed_dict={
                x_: x,
                y_: y,
                batch_size: 333
            })
            if step % 50 == 0:
                # print(
                #     sess.run(cost, feed_dict={
                #         x_: test_data_1,
                #         y_: test_label,
                #         batch_size: 222})
                #     # sess.run(weights['out'])
                #     )
                # print('======================================')
                cost_value = sess.run(cost, feed_dict={
                        x_: x,
                        y_: y,
                        batch_size: 333})
                accu_value = sess.run(accuracy, feed_dict={
                    x_: test_data_3,
                    y_: test_label,
                    batch_size: 222})
                accuracy_file.write(str(step) + ',' + str(accu_value) + '\n')
                cost_file.write(str(step) + ',' + str(cost_value) + '\n')
        accuracy_file.close()
        cost_file.close()


def run_lstm_concat():
    with tf.Session() as sess:
        accuracy_file = open('accuracy_file_concat.csv', 'w')
        accuracy_file.write('step,accuracy\n')
        cost_file = open('cost_file_concat.csv', 'w')
        cost_file.write('step,cost\n')
        sess.run(init)

        x, y = list(train_data_1) + list(train_data_2) + list(train_data_3), train_label + train_label + train_label
        test_data_concat, test_label_concat = \
            list(test_data_1) + list(test_data_2) + list(test_data_3), test_label + test_label + test_label
        for step in range(0, TRAIN_STEP):
            sess.run([train_op], feed_dict={
                x_: x,
                y_: y,
                batch_size: 999
            })
            if step % 50 == 0:
                # print(
                #     sess.run(cost, feed_dict={
                #         x_: test_data_1,
                #         y_: test_label,
                #         batch_size: 222})
                #     # sess.run(weights['out'])
                #     )
                # print('======================================')
                cost_value = sess.run(cost, feed_dict={
                        x_: x,
                        y_: y,
                        batch_size: 999})
                accu_value = sess.run(accuracy, feed_dict={
                    x_: test_data_concat,
                    y_: test_label_concat,
                    batch_size: 666})
                accuracy_file.write(str(step) + ',' + str(accu_value) + '\n')
                cost_file.write(str(step) + ',' + str(cost_value) + '\n')
        accuracy_file.close()
        cost_file.close()
