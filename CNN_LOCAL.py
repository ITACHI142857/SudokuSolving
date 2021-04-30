# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 14:13
# @Author  : YangXiao
# @Project : 读取 MNIST 数据集的训练模型，然后使用本地数据集继续训练模型，并保存
# @Software: PyCharm

from sklearn.model_selection import train_test_split
from imutils import paths
import random
import cv2
import os
import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils


print("loading images")
data = []
labels = []

# 列出路径下的所有文件名并存入 list 列表，以便 for 循环时使用
imagePaths = sorted(list(paths.list_images(".\\digits\\")))

# 固定随机种子，使具有可重复性
random.seed(42)

# 洗牌
random.shuffle(imagePaths)

# 循环遍历所有图片
for imagePath in imagePaths:
    img = cv2.imread(imagePath, 0)

    # 图片大小预处理
    img_resize = cv2.resize(img, (28, 28))
    img_resize = img_resize.reshape(28, 28, 1)

    # 保存图片数据
    data.append(img_resize)

    # 从图像路径中提取分类标签并存储到分类标签数组中
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# 数据归一化
data = np.array(data, dtype="float") / 255.0

# 标签转化为array数组
labels = np.array(labels)

# 构建训练和测试数据集
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
                                                      test_size=0.25, random_state=42)

# 将标签转化为二值序列
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 获取类别数量
num_classes = Y_test.shape[1]


# 读取预训练模型
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# 读取预训练权重参数
loaded_model.load_weights("model.h5")

print("Loaded saved model from disk.")

# 编译模型，指定损失函数为 categorical_crossentropy，优化器为 adam，模型评估标准为 accuracy
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型训练，传入训练集，验证集，指定 epochs 和 batch_size
loaded_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)

# - - - - - - - 保存模型 - - - - - - - -

# 把模型保存到 JSON 文件中
model_json = loaded_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# 把权重参数保存到 HDF5文件中
loaded_model.save_weights("model.h5")

print("Saved model to disk")