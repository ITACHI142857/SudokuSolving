import numpy as np


# 检验数字 n 是否能填在(y,x)位置
def possible(sudoku, y, x, n):
    for i in range(9):
        if sudoku[y][i] == n:
            return False
    for i in range(9):
        if sudoku[i][x] == n:
            return False
    x0 = (x // 3) * 3
    y0 = (y // 3) * 3
    for i in range(3):
        for j in range(3):
            if sudoku[y0 + i][x0 + j] == n:
                return False
    return True


# 解数独
def solve(sudoku):
    for y in range(9):
        for x in range(9):
            if sudoku[y][x] == 0:
                for n in range(1, 10):
                    if possible(sudoku, y, x, n):
                        sudoku[y][x] = n
                        solved = solve(sudoku)
                        if solved is not None:
                            return solved
                        sudoku[y][x] = 0
                return
    return sudoku


# # 测试
# grid = [[0, 6, 0, 7, 0, 9, 0, 4, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 7, 4, 0, 0, 9, 0, 8],
#         [3, 0, 0, 0, 0, 0, 5, 0, 4],
#         [0, 4, 0, 0, 0, 0, 0, 9, 0],
#         [1, 0, 9, 0, 0, 0, 0, 0, 2],
#         [8, 0, 1, 0, 0, 4, 7, 0, 0],
#         [0, 0, 0, 0, 2, 0, 0, 0, 0],
#         [0, 2, 0, 9, 0, 6, 0, 5, 0]]

# print(np.matrix(solve(grid)))
