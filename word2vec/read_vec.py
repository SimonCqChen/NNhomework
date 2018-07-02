import json
import numpy as np
import pymysql

DIMENSION = 300
SE_VALID = ['用例图', '活动图', '敏捷开发', '基线', '需求', '构件', '泛化', '里程碑']


def cosine_dist(list1, list2):
    vector1 = np.array(list1)
    vector2 = np.array(list2)

    op7 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return op7


db = pymysql.connect(
    host='202.120.40.28',
    port=33060,
    user='admin',
    passwd='se1405lab',
    db='word_embeddings'
)

# 使用cursor()方法获取操作游标
cursor = db.cursor()
#
#
# f = open("D:\文档\sgns.zhihu.bigram-char", "r", encoding='UTF-8')
# embeddings = dict()
# index = 1
# for line in f:
#     elements = line.split(' ')
#     if '\n' in elements:
#         elements.remove('\n')
#     if len(elements) == DIMENSION + 1:
#         cur_embedding = [float(elements[i + 1]) for i in range(0, DIMENSION)]
#         try:
#             sql = "INSERT INTO embeddings(word, embedding) VALUES ('%s', '%s')" % (elements[0], str(cur_embedding))
#             cursor.execute(sql)
#         except:
#             print("save failed")
#             continue
#         # embeddings[elements[0]] = cur_embedding
#
#     if index % 1000 == 0:
#         print('embedding read', index)
#     index += 1
#
# # for word in embeddings:
# #     sql = "INSERT INTO embeddings(word, embedding) VALUES ('%s', '%s')" % (word, str(embeddings[word]))
# #     cursor.execute(sql)
#
# db.commit()
# db.close()
# f.close()

for valid_word in SE_VALID:
        if valid_word not in embeddings:
            print(valid_word, 'not in the embeddings')
        else:
            cosine_sim = dict()
            for word in embeddings:
                try:
                    cosine_sim[word] = cosine_dist(embeddings[valid_word], embeddings[word])
                except:
                    continue
            ranked_sim = sorted(cosine_sim.items(), key=lambda item: item[1])
            print(valid_word, ':', ranked_sim[:8])


