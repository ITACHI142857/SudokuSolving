import os
import cv2
from SudokuExtractor import extract_sudoku
from SudokuSolver import solve
from NumberExtractor import extract_number, display_sudoku
import numpy as np

# 读图
filePath = ".\\imgs\\"
files = os.listdir(filePath)
for file in files:
    path = filePath + file

    image_ori = cv2.imread(path, 0)
    ratio = image_ori.shape[0] / 540
    w = round(image_ori.shape[1] / ratio)
    image_resize = cv2.resize(image_ori, (w, 540))

    image = extract_sudoku(path)
    image_show = cv2.resize(image, (540, 540))

    result = extract_number(image)
    result = solve(result)
    if result is not None:
        print("Solution is:")
        display_sudoku(result)
    else:
        print("No Solution!")

    image_stack = np.hstack((image_resize, image_show))

    cv2.imshow("image", image_stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
