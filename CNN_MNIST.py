import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# 设置图片数据格式为 "channels_last"，即 NHWC
K.set_image_data_format('channels_last')

# 固定随机种子，使具有可重复性
seed = 7
numpy.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 把数据集变为 NHWC 格式
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# 数据归一化 from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# 将标签转化为二值序列
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 获取类别数量
num_classes = Y_test.shape[1]

# 创建模型
model = Sequential()
# 2个卷积层
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout层
model.add(Dropout(0.2))
# Flatten层
model.add(Flatten())
# 3个Dense层
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型，指定损失函数为 categorical_crossentropy，优化器为 adam，模型评估标准为 accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型训练，传入训练集，验证集，指定 epochs 和 batch_size
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200)

# 评估模型
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

# - - - - - - - 保存模型 - - - - - - - -

# 把模型保存到 JSON 文件中
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# 把权重参数保存到 HDF5文件中
model.save_weights("model.h5")

print("Saved model to disk")
