from SudokuExtractor import extract_sudoku
import cv2
import numpy as np
from keras.models import model_from_json
import sys
import os

# 读取训练模型
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# 读取训练权重参数
loaded_model.load_weights("model.h5")

print("Loaded saved model from disk.")


# 识别数字图片
def identify_number(img):
    # 图片大小处理
    img_show = cv2.resize(img, (28, 28))
    img_resize = img_show.reshape(1, 28, 28, 1)
    # 把图片输入模型进行预测
    loaded_model_pred = np.argmax(loaded_model.predict(img_resize), axis=-1)
    # 返回预测值
    return loaded_model_pred[0]


# 提取数字图片
def extract_number(sudoku):
    # 把图片大小转为450 * 450，然后以50 * 50切分为81张图片进行识别
    sudoku = cv2.resize(sudoku, (450, 450))
    grid = np.zeros([9, 9])
    for i in range(9):
        for j in range(9):
            img = sudoku[i * 50:(i + 1) * 50, j * 50:(j + 1) * 50]
            '''
            如果像素总值大于 100000，那么认为图片中有数字；
            100000的由来和图片大小450 * 450有关，是经过本地测试得到的；
            有数字的图片像素总值总是大于 100000，无数字的图片像素总值总是小于 100000；
            '''
            if img.sum() > 100000:
                grid[i][j] = identify_number(img)
            else:
                grid[i][j] = 0
    return grid.astype(int)


def output(a):
    sys.stdout.write(str(a))


# 打印数独
def display_sudoku(sudoku):
    output("--------+----------+---------\n")
    for i in range(9):
        for j in range(9):
            cell = sudoku[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')

            if j != 8:
                output('  ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("--------+----------+---------\n")
    output("--------+----------+---------\n")
    output("\n")


# # 读图
# filePath = ".\\imgs\\"
# files = os.listdir(filePath)
# for file in files:
#     path = filePath + file
#     image = extract_sudoku(path)
#     result = extract_number(image)
#     display_sudoku(result)
