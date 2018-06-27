import os
import collections
import re
import pickle
import tensorflow as tf


def reverse_dic(curDic):
  newmaplist = {}
  for key, value in curDic.items():
    newmaplist[value] = key
  return newmaplist


def build_vocab(data):
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def update_char_dic(char_dic, char):
  if char in char_dic:
    char_dic[char] += 1
  else:
    char_dic[char] = 1


def read_words(data_pah):
  chars = []
  with open(data_pah) as f:
    lines = f.readlines()
    for i in range(len(lines)):
      line = lines[i]
      if len(line) == 1:
        # FIXME 这里在可能会出bug，但是暂时不管了
        try:
          nextline = lines[i + 1]
          if len(nextline) == 1:
            chars.append('EoC')
          else:
            continue
        except:
          continue
      if len(re.findall('^第.+章', line)) > 0:
        chars.append('BoC')
      for char in line[:-1]:
        chars.append(char)
      if line[-1] == '\n':
        chars.append('EoL')

  return chars


def data_to_ids(data, word_to_id):
  return [word_to_id[word] for word in data if word in word_to_id]


def read_data(data_path, saving_folder):
  # word_to_id
  word_to_id_path = os.path.join(saving_folder, 'word_to_id.pkl')
  if not os.path.exists(word_to_id_path):
    data = read_words(data_path)
    word_to_id = build_vocab(data)
    pickle.dump(word_to_id,
                open(word_to_id_path, 'wb'),
                True)
  else:
    word_to_id = pickle.load(open(word_to_id_path, 'rb'))

  # ids
  type_set = ['train', 'valid', 'test']
  need_build_file = False
  for type in type_set:
    path = os.path.join(saving_folder, type + '_ids.pkl')
    if not os.path.exists(path):
      need_build_file = True
      break

  if not need_build_file:
    data_ids_list = [pickle.load(open(os.path.join(saving_folder, type + '_ids.pkl'), 'rb')) for type in type_set]
    return data_ids_list[0], data_ids_list[1], data_ids_list[2], len(word_to_id), word_to_id

  # fixme 这里可能会重复读data，暂时不管他
  data = read_words(data_path)

  word_to_id = build_vocab(data)
  vocab_size = len(word_to_id)

  train_data = data[:int(len(data) * 0.8)]
  valid_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
  test_data = data[int(len(data) * 0.9):]

  train_data_ids = data_to_ids(train_data, word_to_id)
  valid_data_ids = data_to_ids(valid_data, word_to_id)
  test_data_ids = data_to_ids(test_data, word_to_id)

  data_ids_list = [train_data_ids, valid_data_ids, test_data_ids]
  for i in range(len(type_set)):
    type = type_set[i]
    data_ids = data_ids_list[i]
    pickle.dump(data_ids,
                open(os.path.join(saving_folder, type + '_ids.pkl'), 'wb'),
                True)

  return train_data_ids, valid_data_ids, test_data_ids, vocab_size, word_to_id


def data_producer(raw_data, batch_size, num_steps, name=None):
  with tf.name_scope(name, "DataProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    # 数据取整，最后多余的一点就扔掉了
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
      epoch_size,
      message="epoch_size == 0, decrease batch_size or num_steps")
    # TODO tf.identity??
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=True).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
