import tensorflow as tf
import numpy as np
import heapq

from char_rnn.src import data_reader

embedding_size = 25
hidden_size = 512
rnn_layers = 2
keep_prob = 1
vocab_size = 65

batch_size = 100
num_steps = 100

learning_rate = 0.005
init_scale = 0.1

MAX_EPOCH = 25
MAX_GRAD_NORM = 5  # 最大梯度，用于梯度修剪


class RNNModel:
    def __init__(self, is_training, batch_size, num_steps):

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 构建cell
        def lstm_cell(hidden_size):
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

        if is_training:
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
            self.train_op = self.build_opt(self.loss, self.learning_rate)

    def build_opt(self, loss, learning_rate):

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_variables), MAX_GRAD_NORM)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        return train_op


def run_epoch(sess, model, train_op, data_queue, total_steps, id_to_word):
    state = sess.run(model._initial_state)
    for step in range(total_steps):
        datas, targets = sess.run(data_queue)

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


def read_shakespeare_data(data_path, saving_folder):
    data = []
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            for char in line:
                if char == '\n':
                    data.append('<EoL>')
                else:
                    data.append(char)

    word_to_id = data_reader.build_vocab(data)

    train_data = data[:int(len(data) * 0.8)]
    valid_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]

    train_data_ids = data_reader.data_to_ids(train_data, word_to_id)
    valid_data_ids = data_reader.data_to_ids(valid_data, word_to_id)
    test_data_ids = data_reader.data_to_ids(test_data, word_to_id)

    return train_data_ids, valid_data_ids, test_data_ids, vocab_size, word_to_id


def pick_ids(preds):
    ids = []
    for pred_per_batch in preds:
        ids_per_batch = []
        for pred in pred_per_batch:
            top_5_prob = heapq.nlargest(5, pred)
            top_5_index = heapq.nlargest(5, range(len(pred)), pred.take)

            s = sum(top_5_prob)
            rnd = np.random.uniform(0, s)

            cur_s = 0
            for p, index in zip(top_5_prob, top_5_index):
                cur_s += p
                if rnd <= cur_s:
                    ids_per_batch.append(index)
                    break
        ids.append(ids_per_batch)
    return ids


def generate_sample():
    data_path = '../data/shakespeare.txt'
    raw_train_data, raw_valid_data, raw_test_data, vocab_size_, word_to_id = read_shakespeare_data(data_path,
                                                                                                   saving_folder='../data')
    id_to_word = data_reader.reverse_dic(word_to_id)

    sample_batch_size = 10
    with tf.variable_scope("Model", reuse=None):
        para_model = RNNModel(is_training=False, batch_size=sample_batch_size, num_steps=1)

    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True
    # with tf.Session(config=sess_config) as sess:

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # saver.restore(sess, '../result/shakeSpeare-[512]-[64+128]/model-6.ckpt')
        saver.restore(sess, '../result/shakeSpeare/model-0.ckpt')

        next_input_char = [[chr(65 + j)] for j in range(sample_batch_size)]
        article = [[next_input_char[j][0]] for j in range(sample_batch_size)]

        state = sess.run(para_model._initial_state)
        for i in range(1000):
            input_id_batch = [[word_to_id[next_input_char[j][0]]] for j in range(sample_batch_size)]
            pred, state = sess.run([para_model.final_prob, para_model.final_state],
                                   {para_model.input_data: input_id_batch, para_model._initial_state: state})
            # id = np.argmax(pred, axis=-1)
            id = pick_ids(pred)

            next_input_char = [[id_to_word[id[j][0]]] for j in range(sample_batch_size)]
            for j in range(sample_batch_size):
                article[j].append(next_input_char[j][0])
        for j in range(sample_batch_size):
            print(''.join(article[j]).replace('<EoL>', '\n'))
            print('=' * 80)
            # embedding, softmax_w, softmax_b = sess.run([para_model.embedding, para_model.softmax_w, para_model.softmax_b])


def main():
    # 读取数据
    data_path = '../data/shakespeare.txt'
    raw_train_data, raw_valid_data, raw_test_data, vocab_size_, word_to_id = read_shakespeare_data(data_path,
                                                                                                   saving_folder='../data')
    vocab_size = vocab_size_

    print(len(raw_train_data))
    print(len(raw_valid_data))
    # 处理成
    train_data_queue = data_reader.data_producer(raw_train_data, batch_size, num_steps)
    valid_data_queue = data_reader.data_producer(raw_valid_data, batch_size, num_steps)

    id_to_word = data_reader.reverse_dic(word_to_id)

    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.name_scope('Train'):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_model = RNNModel(is_training=True, batch_size=batch_size, num_steps=num_steps)

    # with tf.name_scope('Valid'):
    #   # 注意这里的variable_scope要和训练集一致，而且train的reuse=None而valid和test的reuse都是Ture
    #   with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #     valid_model = RNNModel(is_training=False, batch_size=batch_size, num_steps=num_steps)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(MAX_EPOCH):
            print('===========[%d]training===========' % i)
            run_epoch(sess, train_model, train_model.train_op,
                      data_queue=train_data_queue,
                      total_steps=(len(raw_train_data) // batch_size - 1) // num_steps,
                      id_to_word=id_to_word)
            saver.save(sess, '../result/shakeSpeare/model-%d.ckpt' % int(i / 10))

            # print('===========[%d]validation===========' % i)
            # run_epoch(sess, valid_model, tf.no_op(),
            #           data_queue=valid_data_queue,
            #           total_steps=(len(raw_valid_data) // batch_size - 1) // num_steps,
            #           id_to_word=id_to_word)
        coord.request_stop()
        coord.join(threads)
        # run_rnn_epoch(sess,test_model)


if __name__ == '__main__':
    # main()
    generate_sample()
