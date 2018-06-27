import tensorflow as tf
import numpy as np
from char_rnn.src import data_reader

embedding_size = 50
hidden_size = 512
rnn_layers = 2
keep_prob = 0.5
vocab_size = 4060

batch_size = 64
num_steps = 100

learning_rate = 0.001
init_scale = 0.1

MAX_EPOCH = 51
MAX_GRAD_NORM = 5  # 最大梯度，用于梯度修剪


class RNNModel:
    def __init__(self, is_training, batch_size, num_steps, is_test=False):

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 构建cell
        def lstm_cell(hidden_size):
            # FIXME: forget_bias=0.0, state_is_tuple=True如果不加这个，test时会出错
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
            if is_training and keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(hidden_size) for _ in range(rnn_layers)])

        # 查找wordVector
        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        self._initial_state = cell.zero_state(batch_size, tf.float32)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        self.final_state = state

        outputs = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        self.softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
        self.softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        self.logits = tf.matmul(outputs, self.softmax_w) + self.softmax_b
        self.final_prob = tf.nn.softmax(self.logits)
        self.final_prob = tf.reshape(self.final_prob, [batch_size, num_steps, -1])

        if not is_test:
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * num_steps], dtype=tf.float32)])
            # self.loss = tf.reduce_sum(self.loss) / tf.to_float(batch_size)
            self.loss = tf.reduce_mean(self.loss)

            global_step = tf.contrib.framework.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(
                learning_rate,
                global_step,
                50,
                0.98
            )

            # 控制梯度大小，定义优化方法和训练步骤。
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), MAX_GRAD_NORM)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


def run_epoch(sess, model, train_op, data_queue, total_steps):
    state = sess.run(model._initial_state)
    for step in range(total_steps):
        datas, targets = sess.run(data_queue)

        # 这里加入state，让前后的state循环利用。
        _, loss_, state = sess.run([train_op, model.loss, model.final_state],
                                   {model.input_data: datas, model.targets: targets, model._initial_state: state})
        if step % 500 == 0:
            final_prob = sess.run(model.final_prob,
                                  {model.input_data: datas, model.targets: targets})
            pred_ids = np.argmax(final_prob, axis=-1)
            for i in range(1):
                target_sentence = ''.join([id_to_word[id] for id in targets[i]])
                pred_sentence = ''.join([id_to_word[id] for id in pred_ids[i]])
                print('T:%s\nP:%s' % (target_sentence, pred_sentence))
            print(loss_)


if __name__ == '__main__':
    # 读取数据
    data_path = '../data/《斗破苍穹》全本.txt'
    raw_train_data, raw_valid_data, raw_test_data, vocab_size, word_to_id = data_reader.read_data(data_path,
                                                                                                  saving_folder='../data')

    print(len(raw_train_data))
    print(len(raw_valid_data))
    # 处理成
    train_data_queue = data_reader.data_producer(raw_train_data, batch_size, num_steps)
    valid_data_queue = data_reader.data_producer(raw_valid_data, batch_size, num_steps)

    id_to_word = data_reader.reverse_dic(word_to_id)

    # 这几个name_scope不能写在sess里面，不然会报参数未初始化错
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.name_scope('Train'):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_model = RNNModel(is_training=True, batch_size=batch_size, num_steps=num_steps)

    with tf.name_scope('Valid'):
        # 注意这里的variable_scope要和训练集一致，而且train的reuse=None而valid和test的reuse都是Ture
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_model = RNNModel(is_training=False, batch_size=batch_size, num_steps=num_steps)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()

        for i in range(MAX_EPOCH):
            print('===========training===========')
            run_epoch(sess, train_model, train_model.train_op,
                      data_queue=train_data_queue,
                      total_steps=(len(raw_train_data) // batch_size - 1) // num_steps)
            saver.save(sess, '../result/doupo/model-%d.ckpt' % int(i / 10))

            print('===========validation===========')
            run_epoch(sess, valid_model, tf.no_op(),
                      data_queue=valid_data_queue,
                      total_steps=(len(raw_valid_data) // batch_size - 1) // num_steps)
        coord.request_stop()
        coord.join(threads)
        # run_rnn_epoch(sess,test_model)
