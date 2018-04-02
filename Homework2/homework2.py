from sklearn import svm, datasets
import numpy as np
import os

data_folder = "F:\\文档\\NN\\data_hw2"
train_data = np.load(os.path.join(data_folder, 'train_data.npy'))
train_label = np.load(os.path.join(data_folder, 'train_label.npy'))
test_data = np.load(os.path.join(data_folder, 'test_data.npy'))
test_label = np.load(os.path.join(data_folder, 'test_label.npy'))
train_label = np.array([int(i) for i in train_label])
test_label = np.array([int(i) for i in test_label])

train_label_positive = np.array([int(i == 1) for i in train_label])
train_label_indifference = np.array([int(i == 0) for i in train_label])
train_label_negative = np.array([int(i == -1) for i in train_label])

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# for i in range(0, 30):
#     print('%f, %f, %f, %f', train_label[i * 100], train_label_positive[i * 100],
#           train_label_indifference[i * 100], train_label_negative[i * 100])

# origin_svm = svm.SVC(kernel='poly')
# origin_svm.fit(train_data, train_label)
# print(test_label[0:5], origin_svm.predict(test_data[0:5]))
svm_positive = svm.SVC(kernel='rbf', probability=True)
svm_indifference = svm.SVC(kernel='rbf', probability=True)
svm_negative = svm.SVC(kernel='rbf', probability=True)

svm_positive.fit(train_data, train_label_positive)
svm_indifference.fit(train_data, train_label_indifference)
svm_negative.fit(train_data, train_label_negative)

pre_y_posi = svm_positive.predict_proba(test_data)
pre_y_indif = svm_indifference.predict_proba(test_data)
pre_y_nega = svm_negative.predict_proba(test_data)

correct_predict = 0
for i in range(0, test_label.size):
    predict = 0
    if pre_y_posi[i][1] > pre_y_nega[i][1] and pre_y_posi[i][1] > pre_y_indif[i][1]:
        predict = 1
    elif pre_y_indif[i][1] > pre_y_nega[i][1] and pre_y_indif[i][1] > pre_y_posi[i][1]:
        predict = 0
    else:
        predict = -1
    if predict == test_label[i]:
        correct_predict += 1
print(correct_predict/test_label.size)



# clf = svm.SVC()
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
#
# #fit()训练
# clf.fit(X,y)
# #predict()预测
# pre_y = clf.predict(X[50:55])
# print(pre_y)
# print(y[50:55])
# #导入numpy
# import numpy as np
# test = np.array([[5.1,2.9,1.8,3.6]])
# #对test进行预测
# test_y = clf.predict(test)
# print(test_y)