# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 9:25
# @Author  : YangXiao
# @Project : 图片处理 + 数独提取
# @Software: PyCharm

import os
import cv2
import operator
import numpy as np


# 把单独的数字图片拼成完整的数独
def show_digits(digits, colour=255):
    """
    Shows list of 81 extracted digits in a grid format
    """
    rows = []
    # 给每一个数字添加白色边框
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1,
                                      cv2.BORDER_CONSTANT, None, colour) for img in digits]

    for i in range(9):
        # axis=1 沿着水平方向进行拼接，把数组拼接成单独的9行
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    # axis=0沿着垂直方向进行拼接，把每一行拼在一起
    img = np.concatenate(rows, axis=0)
    return img


# 预处理图片
def pre_process_image(img, skip_dilate=False):
    """
    Uses a blurring function, adaptive thresholding
    and dilation to expose the main features of an image.
    """
    # 高斯滤波
    img_process = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    img_process = cv2.adaptiveThreshold(img_process, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    img_process = cv2.bitwise_not(img_process, img_process)

    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_process = cv2.dilate(img_process, kernel)

    return img_process


# 找到数独的4个边界点
def find_corners(img):
    """
    Finds the 4 extreme corners of the largest contour in the image.
    """
    # OpenCV版本问题，cv2.findContours 版本3返回值有3个，版本4只有2个
    opencv_version = cv2.__version__.split('.')[0]
    if opencv_version == '3':
        _, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    # bottom-right point has the largest (x + y) value
    # top-left has point smallest (x + y) value
    # bottom-left point has smallest (x - y) value
    # top-right point has largest (x - y) value

    """
    方法1：
    """
    # 初始化
    val_br = polygon[0][0][0] + polygon[0][0][1]
    val_tl = polygon[0][0][0] + polygon[0][0][1]
    val_bl = polygon[0][0][0] - polygon[0][0][1]
    val_tr = polygon[0][0][0] - polygon[0][0][1]
    bottom_right = polygon[0][0]
    top_left = polygon[0][0]
    bottom_left = polygon[0][0]
    top_right = polygon[0][0]

    # 寻找最大轮廓的4个边界点坐标
    for i in range(1, len(polygon)):
        val_add = polygon[i][0][0] + polygon[i][0][1]
        val_minus = polygon[i][0][0] - polygon[i][0][1]
        if val_add > val_br:
            val_br = val_add
            bottom_right = polygon[i][0]
        if val_add < val_tl:
            val_tl = val_add
            top_left = polygon[i][0]
        if val_minus < val_bl:
            val_bl = val_minus
            bottom_left = polygon[i][0]
        if val_minus > val_tr:
            val_tr = val_minus
            top_right = polygon[i][0]
    # 返回数独的4个边界点坐标
    return [top_left, top_right, bottom_right, bottom_left]

    # """
    # 方法2
    # """
    # # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
    #
    # bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    #
    # # Return an array of all 4 points using the indices
    # # Each point is in its own array of one coordinate
    # return [polygon[top_left][0], polygon[top_right][0],
    #         polygon[bottom_right][0], polygon[bottom_left][0]]


# 两点间距离
def distance_between(p1, p2):
    """
    Returns the scalar distance between two points
    """
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


# 得到只含数独的正方形
def crop_and_warp(img, crop_rect):
    """
    Crops and warps a rectangular section from an image into a square of similar size.
    """
    # 数独的4个边界点坐标
    top_left, top_right, bottom_right, bottom_left = \
        crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or 'getPerspectiveTransform' will throw an error
    # 原始坐标
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # 得到数独的最长边界
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # 使用数独的最长边界来构建新的坐标
    """
    top_left = [0,0], top_right = [side-1,0]
    bottom_right = [side-1,side-1], bottom_left = [0,side-1]
    """
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    img = cv2.warpPerspective(img, matrix, (round(side), round(side)))

    return img


# 把数独分成81个小正方形
def infer_grid(img):
    """
    Infers 81 cell grid from a square image.
    """
    # 数独图片是正方形，把 边长/9 作为步长来确定数独中每个方框的坐标
    squares = []
    side = img.shape[:1]
    step = side[0] / 9

    # 从左到右，再从上到下
    for j in range(9):
        for i in range(9):
            p1 = (i * step, j * step)  # Top left corner of a bounding box
            p2 = ((i + 1) * step, (j + 1) * step)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares


# 对图片进行尺度变换和居中
def scale_and_centre(img, size, margin=0, background=0):
    """
    Scales and centres an image onto a new background square.
    """
    h, w = img.shape[:2]

    # 获取居中位置
    def centre_pad(length):
        """
        Handles centering for a given length that may be odd or even.
        """
        if length % 2 == 0:
            side1 = round((size - length) / 2)
            side2 = side1
        else:
            side1 = round((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return round(r * x)

    if h > w:
        t_pad = round(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = round(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    # 给数字添加黑色边框
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad,
                             cv2.BORDER_CONSTANT, None, background)
    img = cv2.resize(img, (size, size))

    return img


# 寻找数字
def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
    """
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    # 最大区域和起始像素点
    max_area = 0
    seed_point = (None, None)

    # 如果没有传入左上角topleft和右下角bottomright的值，则分别初始化为[0,0]和[w,h]
    if scan_tl is None:
        scan_tl = [0, 0]
    if scan_br is None:
        scan_br = [width, height]

    # 遍历topleft 到 bottomright,记录搜索出来的区域最大值
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                # 通过cv2.floodfill填充图片中的区域为64，并返回填充区域(area[0]是区域面积)
                # 同时记录最大区域的起始像素点
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)
    # 把所有像素值为255的区域填充为64
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    # 初始化mask
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # 把数字区域填充为255
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, None, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            # 把其他区域填充为0
            if img.item(y, x) == 64:
                cv2.floodFill(img, mask, (x, y), 0)

            # 寻找数字的bbox
            if img.item(y, x) == 255:
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y
                if x < left:
                    left = x
                if x > right:
                    right = x

    bbox = [[left, top], [right, bottom]]
    return bbox


# 裁剪图片
def cut_from_rect(img, rect):
    """
    Cuts a rectangle from an image using the top left and bottom right points.
    """
    return img[round(rect[0][1]):round(rect[1][1]), round(rect[0][0]):round(rect[1][0])]


# 处理单个数字
def extract_digit(img, rect, size):
    """
    Extracts a digit (if one exists) from a Sudoku square.
    """
    # 根据81个格子的坐标粗切分出数字
    digit = cut_from_rect(img, rect)

    # Use fill feature finding to get the largest feature in middle of the box
    # m used to define an area in the middle we would expect to find a pixel belonging to the digit

    """
    1、问题：如果从topleft [0,0]开始搜索，那么边缘区域面积可能大于数字区域面积，会把边缘区域误识别为数字区域
    2、解决方案：因为数独中的数字一般都位于格子的中间位置，所以从topleft [m,m]一直搜索到bottomright[w-m,h-m]，
    从而避开边缘区域，这样搜索到的最大面积的区域一定是数字区域
    """
    h, w = digit.shape[:2]
    m = round(np.mean([h, w]) / 2.5)
    bbox = find_largest_feature(digit, [m, m], [w - m, h - m])

    # 使用获取的数字区域的精确 bbox 重新切分出数字
    digit = cut_from_rect(digit, bbox)

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


# 获取所有数字
def get_digits(img, squares, size):
    """
    Extracts digits from their cells and builds an array
    """
    digits = []
    img = pre_process_image(img.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


# 主程序
def extract_sudoku(img_path):
    img_ori = cv2.imread(img_path, 0)
    # 预处理图片
    img_process = pre_process_image(img_ori)
    # 找到图片中数组的4个边界点
    corners = find_corners(img_process)
    # 根据边界点把原图变为只含数独的正方形
    img_crop = crop_and_warp(img_ori, corners)
    # 得到数独中每个方格的坐标
    squares = infer_grid(img_crop)
    # 获取精确的数字图片
    digits = get_digits(img_crop, squares, 28)
    # 拼接所有数字
    final_image = show_digits(digits, 255)

    return final_image


# 读图
# filePath = ".\\imgs\\"
# files = os.listdir(filePath)
# for file in files:
#     path = filePath + file
#     image = extract_sudoku(path)
#     cv2.imshow(file, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
