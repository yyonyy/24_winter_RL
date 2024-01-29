import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy as dc


# prev_positions = deque(maxlen=3)

# prev_positions.append([0,1])
# prev_positions.append([2,3])
# prev_positions.append([4,5])

# print(prev_positions[2][0])

# for position in prev_positions:
#     plt.scatter(position[0], position[1], c='red')

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Previous Positions')
# plt.grid(True)
# plt.show()0


# grid = np.zeros((200,400,3))

# cv2.line(grid, (0,0),(400,200), (1, 0, 0), 1)

# # # (0, 0) 위치에 빨간색 점 찍기
# # grid[0, 0] = [1, 0, 0]  # BGR 순서이므로 빨간색은 (0, 0, 255)

# # # (1, 0) 위치에 파란색 점 찍기
# # grid[1, 0] = [255, 0, 0]  # BGR 순서이므로 파란색은 (255, 0, 0)

# grid_extra = dc(grid)

# cv2.line(grid_extra, (1,0),(401,200), (255, 0, 0), 1)

# print(grid_extra[0,0][0])

# print(grid_extra[1,0][0])

# print(grid_extra[0,1][0])

# print(grid[0,0,1])

# cv2.imshow("a",grid)
# cv2.imshow("b",grid_extra)
# cv2.waitKey()