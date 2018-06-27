import tensorflow as tf
import numpy as np
import heapq

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
      # FIXME: forget_bias=0.0, state_is_tuple=True如果不加這個，那麼test时会出错
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


def run_epoch(sess, model, data_queue, total_steps, id_to_word=None):
  pred_words = []
  for i in range(5):
    pred_words.append([])

  embedding = sess.run(model.embedding)
  for step in range(total_steps):
    datas, targets = sess.run(data_queue)

    loss_, final_prob = sess.run([model.loss, model.final_prob],
                                 {model.input_data: datas, model.targets: targets})

    pred_ids = np.argmax(final_prob, axis=-1)
    for i in range(5):
      target_sentence = ''.join([id_to_word[id] for id in targets[i]])
      pred_sentence = ''.join([id_to_word[id] for id in pred_ids[i]])
      print('T:%s\nP:%s' % (target_sentence, pred_sentence))

      if step % 50 == 0:
        print(loss_)
        for i in range(5):
          print(''.join(pred_words[i]))


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


class BeamSearchUnit:
  def __init__(self, cur_str, prob_list):
    self.cur_str = cur_str
    self.next_char = cur_str[-1]
    # self.state = state
    # if len(prob_list) > 10:
    #   prob_list = prob_list[-10:]
    self.prob = prob_list
    # self.prob = sum(self.prob_list)

  def set_state(self, c_state, h_state):
    self.c_state = c_state
    self.h_state = h_state

  def __lt__(self, other):
    return self.prob < other.prob


def topK_heapq(units, k):
  array = []
  for i in range(len(units)):
    if len(array) < k:
      heapq.heappush(array, units[i])
    else:
      array_min = array[0]
      # or np.random.random() < 0.2
      if units[i] > array_min:
        heapq.heapreplace(array, units[i])
  topK = array
  return topK


def generate_sample():
  data_path = '../data/《斗破苍穹》全本.txt'
  raw_train_data, raw_valid_data, raw_test_data, vocab_size_, word_to_id = data_reader.read_data(data_path,
                                                                                                 saving_folder='../data')
  id_to_word = data_reader.reverse_dic(word_to_id)

  sample_batch_size = 3
  with tf.variable_scope("Model", reuse=None):
    para_model = RNNModel(is_training=False, batch_size=sample_batch_size, num_steps=1, is_test=True)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    saver.restore(sess, '../result/doupo/model-2.ckpt')

    # region #... beam search
    # beam_search_k = 15
    # bs_units = [[BeamSearchUnit('第', 1.0)] * beam_search_k,  # 重复k次, 作为满足Model的batchsize必须固定问题的临时解决方案
    #             [BeamSearchUnit('萧', 1.0)] * beam_search_k,
    #             [BeamSearchUnit('天', 1.0)] * beam_search_k]
    # input_state = sess.run(para_model._initial_state)
    # for i in range(len(bs_units)):
    #   units = bs_units[i]
    #   for j in range(len(units)):
    #     unit = units[j]
    #
    #     c_state = []
    #     h_state = []
    #     for layer in range(2):
    #       cur_c_state = input_state[layer].c[i * len(bs_units) + j]
    #       cur_h_state = input_state[layer].h[i * len(bs_units) + j]
    #
    #       c_state.append(cur_c_state)
    #       h_state.append(cur_h_state)
    #
    #     unit.set_state(c_state, h_state)
    #
    # for i in range(5):
    #   bs_units = [topK_heapq(units, beam_search_k) for units in bs_units]
    #
    #   states = []
    #   for layer in range(2):
    #     c_state = []
    #     h_state = []
    #     for units in bs_units:
    #       for unit in units:
    #         c_state.append(unit.c_state[layer])
    #         h_state.append(unit.h_state[layer])
    #     state = tf.nn.rnn_cell.LSTMStateTuple(
    #       c=np.array(c_state),
    #       h=np.array(h_state))
    #     states.append(state)
    #
    #   input_state = tuple(states)
    #
    #   input_id_batch = []
    #   for units in bs_units:
    #     for unit in units:
    #       input_id_batch.append([word_to_id[unit.next_char]])
    #
    #   preds, input_state = sess.run([para_model.final_prob, para_model.final_state],
    #                                 {para_model.input_data: input_id_batch, para_model._initial_state: input_state})
    #
    #   if i == 0:
    #     for j in range(len(bs_units)):
    #       pred_per_batch = preds[j * beam_search_k]
    #
    #       c_state = []
    #       h_state = []
    #       for layer in range(2):
    #         cur_c_state = input_state[layer].c[j * beam_search_k]
    #         cur_h_state = input_state[layer].h[j * beam_search_k]
    #
    #         c_state.append(cur_c_state)
    #         h_state.append(cur_h_state)
    #
    #       new_units = []
    #       for pred in pred_per_batch:
    #         top_k_prob = heapq.nlargest(beam_search_k, pred)
    #         top_k_index = heapq.nlargest(beam_search_k, range(len(pred)), pred.take)
    #
    #         for k in range(beam_search_k):
    #           # 在i==0时是特例，全都取第0个
    #           ori_unit = bs_units[j][0]
    #           new_unit = BeamSearchUnit(ori_unit.cur_str + id_to_word[top_k_index[k]],
    #                                     ori_unit.prob + top_k_prob[k])
    #           new_unit.set_state(c_state, h_state)
    #           new_units.append(new_unit)
    #       bs_units[j] = new_units
    #   else:
    #     for j in range(len(bs_units)):
    #       new_units = []
    #       for ori_unit, k in zip(bs_units[j], range(len(bs_units[j]))):
    #         pred_per_batch = preds[j * beam_search_k + k]
    #         c_state = []
    #         h_state = []
    #         for layer in range(2):
    #           cur_c_state = input_state[layer].c[j * beam_search_k + k]
    #           cur_h_state = input_state[layer].h[j * beam_search_k + k]
    #
    #           c_state.append(cur_c_state)
    #           h_state.append(cur_h_state)
    #
    #         for pred in pred_per_batch:
    #           top_k_prob = heapq.nlargest(beam_search_k, pred)
    #           top_k_index = heapq.nlargest(beam_search_k, range(len(pred)), pred.take)
    #
    #           for kk in range(beam_search_k):
    #             # 在i==0时是特例，全都取第0个
    #             new_unit = BeamSearchUnit(ori_unit.cur_str + id_to_word[top_k_index[kk]],
    #                                       ori_unit.prob + top_k_prob[kk])
    #             new_unit.set_state(c_state, h_state)
    #             new_units.append(new_unit)
    #       bs_units[j] = new_units
    #
    # bs_units = [topK_heapq(units, 1) for units in bs_units]
    # for units in bs_units:
    #   for unit in units:
    #     print(unit.cur_str.replace('EoL', '\n'))
    #   print('=' * 80)
    # regionend

    # region ...普通概率生成
    next_input_char = [['第'], ['萧'], ['天']]
    article = [[next_input_char[j][0]] for j in range(sample_batch_size)]

    state = sess.run(para_model._initial_state)
    for i in range(1000):
      input_id_batch = [[word_to_id[next_input_char[j][0]]] for j in range(sample_batch_size)]
      pred, state = sess.run([para_model.final_prob, para_model.final_state],
                             {para_model.input_data: input_id_batch, para_model._initial_state: state})
      id = pick_ids(pred)

      next_input_char = [[id_to_word[id[j][0]]] for j in range(sample_batch_size)]
      for j in range(sample_batch_size):
        article[j].append(next_input_char[j][0])
    for j in range(sample_batch_size):
      print(''.join(article[j]).replace('EoL', '\n'))
      print('=' * 80)
      # regionend


def main():
  # 读取数据
  data_path = '../data/《斗破苍穹》全本.txt'
  raw_train_data, raw_valid_data, raw_test_data, vocab_size, word_to_id = data_reader.read_data(data_path,
                                                                                                saving_folder='../data')

  # 处理成
  test_data_queue = data_reader.data_producer(raw_test_data, batch_size, num_steps)

  id_to_word = data_reader.reverse_dic(word_to_id)

  with tf.variable_scope("Model", reuse=None):
    test_model = RNNModel(is_training=False, batch_size=batch_size, num_steps=num_steps)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()
    # saver.restore(sess, '../result/doupo-[1024]-[2]-about3.3-3.5/model-4.ckpt')
    saver.restore(sess, '../result/doupo/model-2.ckpt')
    run_epoch(sess, test_model, test_data_queue,
              total_steps=(len(raw_train_data) // batch_size - 1) // num_steps,
              id_to_word=id_to_word)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  # main()
  generate_sample()
