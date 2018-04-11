import numpy as np
import tensorflow as tf

import struct
import os
import time
import operator
from functools import reduce
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def loadImageSet(which=0):
    print("load image set")
    binfile=None
    if which == 0:
        binfile = open("./train-images.idx3-ubyte", 'rb')
    else:
        binfile = open("./t10k-images.idx3-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII' , buffers ,0)
    print("head,",head)

    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]
    #[60000]*28*28
    bits=imgNum*width*height
    bitsString='>'+str(bits)+'B' #like '>47040000B'

    imgs=struct.unpack_from(bitsString,buffers,offset)

    binfile.close()
    imgs=np.reshape(imgs,[imgNum,width,height])

    data = [image.flatten() for image in imgs]
    # for k in range(0, imgs.shape[0]):
    #     data.append([])
    #     for i in range(0, imgs[k].shape[0]):
    #         for j in range(0, imgs[k].shape[1]):
    #             data[k].append(imgs[k][i][j])
    print("load imgs finished")
    return data

def loadLabelSet(which=0):
    print("load label set")
    binfile=None
    if which==0:
        binfile = open("./train-labels.idx1-ubyte", 'rb')
    else:
        binfile=  open("./t10k-labels.idx1-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II' , buffers ,0)
    print("head,",head)
    imgNum=head[1]

    offset = struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels= struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels=np.reshape(labels,[imgNum,1])

    labels_result = []
    for i in range(0, labels.shape[0]):
        labels_result.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        labels_result[i][labels[i][0]] = 1
    print('load label finished')
    return labels_result


img = loadImageSet()
label = loadLabelSet()
img_test = loadImageSet(1)
label_test = loadLabelSet(1)

sess = tf.Session()

# # 用tensorflow 导入数据
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# # Extracting MNIST_data/train-images-idx3-ubyte.gz
# # Extracting MNIST_data/train-labels-idx1-ubyte.gz
# # Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# # Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# print('training data shape ', mnist.train.images.shape)
# print('training label shape ', mnist.train.labels.shape)
# # training data shape  (55000, 784)
# # training label shape  (55000, 10)

# 权值初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# input_layer
X_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# FC1
W_fc1 = weight_variable([784, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.sigmoid(tf.matmul(X_, W_fc1) + b_fc1)

# FC2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pre = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pre))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

start_time = time.time()

for i in range(5000):
    batch_size = 1500
    start = i * batch_size
    # print(len(img), len(label))
    if start >= len(img):
        start = (i * batch_size) % len(img)
    X_batch, y_batch = img[start: min(len(img), start + batch_size)], label[start: min(len(img), start + batch_size)]

    sess.run(train_step, feed_dict={X_: X_batch, y_: y_batch})
    # print(sess.run(W_fc2))
    if (i+1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={X_: img, y_: label})
        print("step %d, training acc %g" % (i+1, train_accuracy))
        # print(sess.run(W_fc2))
    if (i+1) % 1000 == 0:
        test_accuracy = sess.run(accuracy, feed_dict={X_: img, y_: label})
        print("= " * 10, "step %d, testing acc %g" % (i+1, test_accuracy))
end_time = time.time()
print("time: %s" % str(end_time - start_time))
train_accuracy = sess.run(accuracy, feed_dict={X_: img_test, y_: label_test})
print("final acc %g" % train_accuracy)

