from sklearn import svm
import numpy as np
import os
import random
import time

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


# positive svm divide
posi_1_a_index = []
posi_1_b_index = []
posi_0_a_index = []
posi_0_b_index = []
posi_0_c_index = []
posi_0_d_index = []
for i in range(0, train_label.size):
    if train_label[i] == 0:
        prob = random.random()
        if prob > 0.5:
            posi_0_a_index.append(i)
        else:
            posi_0_b_index.append(i)
    elif train_label[i] == -1:
        prob = random.random()
        if prob > 0.5:
            posi_0_c_index.append(i)
        else:
            posi_0_d_index.append(i)
    elif train_label[i] == 1:
        prob = random.random()
        if prob > 0.5:
            posi_1_a_index.append(i)
        else:
            posi_1_b_index.append(i)

posi_1_a_data = [train_data[i] for i in posi_1_a_index]
posi_1_b_data = [train_data[i] for i in posi_1_b_index]
posi_0_a_data = [train_data[i] for i in posi_0_a_index]
posi_0_b_data = [train_data[i] for i in posi_0_b_index]
posi_0_c_data = [train_data[i] for i in posi_0_c_index]
posi_0_d_data = [train_data[i] for i in posi_0_d_index]

posi_1_a_label = [train_label_positive[i] for i in posi_1_a_index]
posi_1_b_label = [train_label_positive[i] for i in posi_1_b_index]
posi_0_a_label = [train_label_positive[i] for i in posi_0_a_index]
posi_0_b_label = [train_label_positive[i] for i in posi_0_b_index]
posi_0_c_label = [train_label_positive[i] for i in posi_0_c_index]
posi_0_d_label = [train_label_positive[i] for i in posi_0_d_index]

posi_1_data_list = [posi_1_a_data, posi_1_b_data]
posi_1_label_list = [posi_1_a_label, posi_1_b_label]
posi_0_data_list = [posi_0_a_data, posi_0_b_data, posi_0_c_data, posi_0_d_data]
posi_0_label_list = [posi_0_a_label, posi_0_b_label, posi_0_c_label, posi_0_d_label]

# indifferent svm divide
indif_1_a_index = []
indif_1_b_index = []
indif_0_a_index = []
indif_0_b_index = []
indif_0_c_index = []
indif_0_d_index = []
for i in range(0, train_label.size):
    if train_label[i] == 1:
        prob = random.random()
        if prob > 0.5:
            indif_0_a_index.append(i)
        else:
            indif_0_b_index.append(i)
    elif train_label[i] == -1:
        prob = random.random()
        if prob > 0.5:
            indif_0_c_index.append(i)
        else:
            indif_0_d_index.append(i)
    elif train_label[i] == 0:
        prob = random.random()
        if prob > 0.5:
            indif_1_a_index.append(i)
        else:
            indif_1_b_index.append(i)

indif_1_a_data = [train_data[i] for i in indif_1_a_index]
indif_1_b_data = [train_data[i] for i in indif_1_b_index]
indif_0_a_data = [train_data[i] for i in indif_0_a_index]
indif_0_b_data = [train_data[i] for i in indif_0_b_index]
indif_0_c_data = [train_data[i] for i in indif_0_c_index]
indif_0_d_data = [train_data[i] for i in indif_0_d_index]

indif_1_a_label = [train_label_indifference[i] for i in indif_1_a_index]
indif_1_b_label = [train_label_indifference[i] for i in indif_1_b_index]
indif_0_a_label = [train_label_indifference[i] for i in indif_0_a_index]
indif_0_b_label = [train_label_indifference[i] for i in indif_0_b_index]
indif_0_c_label = [train_label_indifference[i] for i in indif_0_c_index]
indif_0_d_label = [train_label_indifference[i] for i in indif_0_d_index]

indif_1_data_list = [indif_1_a_data, indif_1_b_data]
indif_1_label_list = [indif_1_a_label, indif_1_b_label]
indif_0_data_list = [indif_0_a_data, indif_0_b_data, indif_0_c_data, indif_0_d_data]
indif_0_label_list = [indif_0_a_label, indif_0_b_label, indif_0_c_label, indif_0_d_label]

# negative svm divide
nega_1_a_index = []
nega_1_b_index = []
nega_0_a_index = []
nega_0_b_index = []
nega_0_c_index = []
nega_0_d_index = []
for i in range(0, train_label.size):
    if train_label[i] == 1:
        prob = random.random()
        if prob > 0.5:
            nega_0_a_index.append(i)
        else:
            nega_0_b_index.append(i)
    elif train_label[i] == 0:
        prob = random.random()
        if prob > 0.5:
            nega_0_c_index.append(i)
        else:
            nega_0_d_index.append(i)
    elif train_label[i] == -1:
        prob = random.random()
        if prob > 0.5:
            nega_1_a_index.append(i)
        else:
            nega_1_b_index.append(i)

nega_1_a_data = [train_data[i] for i in nega_1_a_index]
nega_1_b_data = [train_data[i] for i in nega_1_b_index]
nega_0_a_data = [train_data[i] for i in nega_0_a_index]
nega_0_b_data = [train_data[i] for i in nega_0_b_index]
nega_0_c_data = [train_data[i] for i in nega_0_c_index]
nega_0_d_data = [train_data[i] for i in nega_0_d_index]

nega_1_a_label = [train_label_negative[i] for i in nega_1_a_index]
nega_1_b_label = [train_label_negative[i] for i in nega_1_b_index]
nega_0_a_label = [train_label_negative[i] for i in nega_0_a_index]
nega_0_b_label = [train_label_negative[i] for i in nega_0_b_index]
nega_0_c_label = [train_label_negative[i] for i in nega_0_c_index]
nega_0_d_label = [train_label_negative[i] for i in nega_0_d_index]

nega_1_data_list = [nega_1_a_data, nega_1_b_data]
nega_1_label_list = [nega_1_a_label, nega_1_b_label]
nega_0_data_list = [nega_0_a_data, nega_0_b_data, nega_0_c_data, nega_0_d_data]
nega_0_label_list = [nega_0_a_label, nega_0_b_label, nega_0_c_label, nega_0_d_label]

svm_list = []
svm_list.append([])
for i in range(0, 2):
    svm_list[0].append([])
    for j in range(0, 4):
        a = time.time()
        current_svm = svm.SVC(kernel='rbf', probability=True)
        current_svm.fit(posi_1_data_list[i] + posi_0_data_list[j], posi_1_label_list[i] + posi_0_label_list[j])
        b = time.time()
        svm_list[0][i].append(current_svm)
        print(b - a)

svm_list.append([])
for i in range(0, 2):
    svm_list[1].append([])
    for j in range(0, 4):
        a = time.time()
        current_svm = svm.SVC(kernel='rbf', probability=True)
        current_svm.fit(indif_1_data_list[i] + indif_0_data_list[j], indif_1_label_list[i] + indif_0_label_list[j])
        b = time.time()
        svm_list[1][i].append(current_svm)
        print(b - a)

svm_list.append([])
for i in range(0, 2):
    svm_list[2].append([])
    for j in range(0, 4):
        a = time.time()
        current_svm = svm.SVC(kernel='rbf', probability=True)
        current_svm.fit(nega_1_data_list[i] + nega_0_data_list[j], nega_1_label_list[i] + nega_0_label_list[j])
        b = time.time()
        svm_list[2][i].append(current_svm)
        print(b - a)


POSI = 0
INDIF = 1
NEGA = 2
A = 0
B = 1
C = 2
D = 3

predict_prob_list = []
for k in range(0, 3):
    predict_prob_list.append([])
    for i in range(0, 2):
        predict_prob_list[k].append([])
        for j in range(0, 4):
            predict_prob_list[k][i].append(svm_list[k][i][j].predict_proba(test_data))

predict_result = []
correct_predict = 0
for test_index in range(0, test_label.size):
    predict = 0
    max_result = []
    for k in range(0, 3):
        min_result = []
        for i in range(0, 2):
            origin_result = []
            for j in range(0, 4):
                origin_result.append(predict_prob_list[k][i][j][test_index][1])
            min_result.append(min(origin_result))
        max_result.append(max(min_result))
    if max_result[POSI] > max_result[INDIF] and max_result[POSI] > max_result[NEGA]:
        predict = 1
    elif max_result[INDIF] > max_result[POSI] and max_result[INDIF] > max_result[NEGA]:
        predict = 0
    else:
        predict = -1

    if predict == test_label[test_index]:
        correct_predict += 1

print(correct_predict/test_label.size)
