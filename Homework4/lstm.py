import numpy as np
import tensorflow as tf

lstm = tf.nn.rnn_cell.BasicLSTMCell

data1 = np.load('./data_used_10399/01.npz')
files_in_zip = data1.keys()
clip1 = data1[files_in_zip[0]]
clip2 = data1[files_in_zip[1]]
label = np.load('./data_used_10399/label.npy')
print(label.shape)
