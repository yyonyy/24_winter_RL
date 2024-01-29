import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.optimize import curve_fit

# 2차 방정식으로 근사
def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

# 3차 방정식으로 근사
# def cubic_function(x, a, b, c, d):
#     return a * x ** 3 + b * x ** 2 + c * x + d

x_coordinates = np.array([19, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 28, 29, 30, 29, 30, 31, 32, 33, 32, 31, 30, 29, 28, 27, 28, 27, 26, 27, 26])
y_coordinates = np.array([39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8])

img = np.zeros((40, 40, 3), dtype=np.uint8)
img_1 = np.zeros((40, 40, 3), dtype=np.uint8)
img_2 = np.zeros((40, 40, 3), dtype=np.uint8)

points = np.column_stack((x_coordinates,y_coordinates))

print(points)

for point in points:
    img[point[1], point[0]] = [255, 255, 0]  #이동 경로
img = np.rot90(img, k=-1)

zero_coordinates = np.argwhere(~np.all(img == [0, 0, 0], axis=-1))

x_coor = zero_coordinates[:, 1]
y_coor = zero_coordinates[:, 0]

# curve_fit 하는 부분
popt, pcov = curve_fit(quadratic_function, x_coor, y_coor)
# popt, pcov = curve_fit(cubic_function, x_coordinates, y_coordinates)

a_opt, b_opt, c_opt = popt
# a_opt, b_opt, c_opt,d_opt = popt

x_fit = np.linspace(min(x_coor), max(x_coor), 100)
y_fit = quadratic_function(x_fit, a_opt, b_opt, c_opt)
# y_fit = cubic_function(x_fit, a_opt, b_opt, c_opt, d_opt)

points_1 = np.column_stack((x_fit,y_fit))

# print(points_1)

for point_1 in points_1:
    img_1[int(point_1[1]), int(point_1[0])] = [255,255,255]

img = np.rot90(img, k=1)
img_1 = np.rot90(img_1, k=1)

# Display the image
cv2.imshow('Point Plot', img)
cv2.imshow("hjihi", img_1)
cv2.waitKey(0)