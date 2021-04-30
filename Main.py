# -*- coding: utf-8 -*-
# @Time    : 2021/4/27 20:01
# @Author  : YangXiao
# @Project : 主程序
# @Software: PyCharm

import os
from SudokuExtractor import extract_sudoku
from SudokuSolver import solve
from NumberExtractor import extract_number, display_sudoku

# 读图
filePath = ".\\imgs\\"
files = os.listdir(filePath)
for file in files:
    path = filePath + file
    image = extract_sudoku(path)
    result = extract_number(image)
    result = solve(result)
    if result is not None:
        print("Solution is:")
        display_sudoku(result)
    else:
        print("No Solution!")