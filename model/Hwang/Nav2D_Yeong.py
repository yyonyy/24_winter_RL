# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:33:29 2019

@author: Or
"""

import torch
import cv2
import numpy as np
import matplotlib as plt
import random
import math
from matplotlib.pyplot import imshow
from copy import deepcopy as dc
from collections import deque

# 변수들

# PIXEL WIDTH
WIDTH = 40
# PIXEL HEIGHT
HEIGHT = 40

# 좌,우 차선 중심점
LEFT_POINT = (9, 19)
RIGHT_POINT = (29, 19)

# 차량의 현재 위치
CURRENT_POS = None

# 장애물 크기 (1 -> 3x3, 2 -> 4x4)
OBSTACLE_SIZE = 1
# 장애물 마진 크기
OBSTACLE_SURPLUS = 1

# 차선 크기
LANE_WIDTH = 1
# 차선 마진 크기
LANE_SURPLUS = 1


class Navigate2D:
    def __init__(self,Nobs,Dobs,Rmin):
        self.W = WIDTH
        self.H = HEIGHT
        self.Nobs = Nobs
        self.Dobs = Dobs
        self.Rmin = Rmin
        self.state_dim = [self.W,self.H,3]
        self.action_dim = 5
        self.scale = 10.0
        self.real_scale = 20.0
        self.consecutive_steps = 0
        self.prev_positions = deque(maxlen=4)
        self.over_lane = 0
        
    def get_dims(self):
        return self.state_dim, self.action_dim
        
    def reset(self):
        # map 밑배경 생성
        grid = np.zeros((self.H,self.W,3))

        # self.slope = math.tan(math.pi * random.randint(45,135)/180)
        self.slope = math.tan(math.pi * 90/180)

        # 차선 생성
        left_start_x = int(LEFT_POINT[0] - LEFT_POINT[0] / self.slope)
        left_start_y = HEIGHT - 1

        left_end_x = int(LEFT_POINT[0] + LEFT_POINT[0] / self.slope)
        left_end_y = 0
        
        right_start_x = left_start_x + int(WIDTH / 2)
        right_start_y = HEIGHT - 1
        
        right_end_x = left_end_x + int(WIDTH / 2)
        right_end_y = 0
        
        self.consecutive_steps = 0
        # self.prev_pos = None
        
        # 차선 draw
        cv2.line(grid, (left_end_x,left_end_y),(left_start_x,left_start_y),
                 (2, 0, 0), LANE_WIDTH)
        
        cv2.line(grid, (right_end_x,right_end_y),(right_start_x, right_start_y), 
                 (2, 0, 0), LANE_WIDTH)

        curr_Nobs = self.Nobs

        # 라바콘 생성
        for _ in range(curr_Nobs):
            # 장애물의 y 중심 생성
            ################ => trigger range 고려해서 수정
            center_y = random.randint(15,27)

            # 선택한 y값 기준 직선 안쪽에 있는 경계들 중에서 랜덤으로 x값 생성
            x_1 , x_2 = (min(np.argwhere(grid[center_y,:,0] == 2))+2)[0], (max(np.argwhere(grid[center_y,:,0] == 2))-2)[0]
            # print("z: ",np.argwhere(grid[center_y,:,0] == 2),"x1 : ",x_1, "x2 : ",x_2)
            center_x = random.randint(min(x_1,x_2), max(x_1,x_2))

            minX = center_x - OBSTACLE_SIZE
            minY = center_y - OBSTACLE_SIZE
            maxX = center_x + OBSTACLE_SIZE
            maxY = center_y + OBSTACLE_SIZE
            grid[minY:maxY+1,minX:maxX+1,0] = 1.0

            # labacorn surplus 생성
            grid[minY-OBSTACLE_SURPLUS:minY,minX-OBSTACLE_SURPLUS:maxX+OBSTACLE_SURPLUS+1,0] = 255.0
            grid[minY:maxY+1,minX-OBSTACLE_SURPLUS:minX,0] = 255.0
            grid[minY:maxY+1,maxX+1:maxX+OBSTACLE_SURPLUS+1,0] = 255.0
            grid[maxY+1:maxY+OBSTACLE_SURPLUS+1,minX-OBSTACLE_SURPLUS:maxX+OBSTACLE_SURPLUS+1,0] = 255.0

        # lane surplus 생성
        cv2.line(grid, (left_end_x-LANE_SURPLUS,left_end_y),(left_start_x-LANE_SURPLUS,left_start_y),
                    (255, 0, 0), LANE_SURPLUS)
        cv2.line(grid, (left_end_x+LANE_SURPLUS,left_end_y),(left_start_x+LANE_SURPLUS,left_start_y),
                    (255, 0, 0), LANE_SURPLUS)
        cv2.line(grid, (right_end_x-LANE_SURPLUS,right_end_y),(right_start_x-LANE_SURPLUS, right_start_y), 
                    (255, 0, 0), LANE_SURPLUS)
        cv2.line(grid, (right_end_x+LANE_SURPLUS,right_end_y),(right_start_x+LANE_SURPLUS, right_start_y), 
                    (255, 0, 0), LANE_SURPLUS)
        
       # 출발점 생성. 차선 안쪽에서 생성하도록
        start = (HEIGHT-1,int((left_start_x+right_start_x)/2))
        # start = (39,random.randint(int((left_start_x+right_start_x)/2)-2,int((left_start_x+right_start_x)/2)+2))
        # finish = (0,int((left_end_x+right_end_x)/2))
        
        finish_range = range(int((left_end_x + right_end_x) / 2) - 3, int((left_end_x + right_end_x) / 2) + 4)
        finish_points = [(0, x) for x in finish_range]
        
        grid[start[0],start[1],1] = self.scale*1.0
        # grid[finish[0],finish[1],2] = self.scale*1.0
        
        for point in finish_points:
            if point[1] == int((left_end_x + right_end_x) / 2):
                grid[point[0], point[1], 2] = self.real_scale*1.0
                grid[point[0]+1, point[1], 2] = self.real_scale*1.0
                grid[point[0]+2, point[1], 2] = self.real_scale*1.0
                # print("point", point[0], point[1], grid[point[0], point[1], 2])
            else:
                grid[point[0], point[1], 2] = self.scale*1.0
                grid[point[0]+1, point[1], 2] = self.scale*1.0
                grid[point[0]+2, point[1], 2] = self.scale*1.0
                # print("1", point[0], point[1], grid[point[0], point[1], 2])
        done = False

        return grid, done
    
    def make_car_boound(self, grid, yaw, pos) :
        # 먼저 차량의 앞, 뒤 코 설정
        car_front = (int(pos[1] + 8*math.cos(yaw)), int(pos[0] - 8*math.sin(yaw)))
        car_rear = (int(pos[1] - 3*math.cos(yaw)), int(pos[0] + 3*math.sin(yaw)))
        # print("pos: ",pos,"yaw: ",yaw,"front : ", car_front, "rear : ",car_rear)

        # 차량 앞, 뒤 boudary 생성
        if yaw == math.pi * 90/180 :
            front_left = (car_front[0]-4,car_front[1])
            front_right = (car_front[0]+4,car_front[1])
            rear_left = (car_rear[0]-4, car_rear[1])
            rear_right = (car_rear[0]+4, car_rear[1])
        else :        
            front_right = (car_front[0]+4*math.sin(yaw),car_front[1]+4*math.cos(yaw))
            front_left = (car_front[0]-4*math.sin(yaw),car_front[1]-4*math.cos(yaw))
            rear_right = (car_rear[0]+4*math.sin(yaw), car_rear[1]+4*math.cos(yaw))
            rear_left = (car_rear[0]-4*math.sin(yaw), car_rear[1]-4*math.cos(yaw))

        points = np.array([front_left, front_right, rear_right, rear_left], np.int32)
        points = points.reshape(-1,1,2)
        car_grid = cv2.fillPoly(grid, [points],[0,255,255])
        # cv2.line(car_grid,(car_front[0],car_front[1]),(car_front[0],car_front[1]),(1,0,1),1)
        # cv2.line(car_grid,(car_rear[0],car_rear[1]),(car_rear[0],car_rear[1]),(1,0,1),1)
        # cv2.line(car_grid,(pos[1],pos[0]),(pos[1],pos[0]),(0,0,1),1)
        # cv2.imshow("C",car_grid)
        # cv2.waitKey(0)
        return car_grid
    
    def det_yaw(self, action) :
        if (action == np.array([-1,0])).all() :
            return math.pi * 90/180
        elif (action == np.array([0,1])).all() :
            return math.pi * 45/180
        elif (action == np.array([-1,1])).all() :
            return math.pi * 60/180
        elif (action == np.array([0,-1])).all() :
            return math.pi * 135/180
        elif (action == np.array([-1,-1])).all() :
            return math.pi * 120/180  
    
    def step(self,grid,action,prev_action):
        prev_position = self.prev_positions
        # max_norm = self.N
        new_grid = dc(grid)
        car_grid = dc(grid)
        reward = -1.0
        
        A = False
        B = False
        
        done = False
        done_1 = False
        crack = False
        # act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        act = np.array([[0,-1],[-1,-1],[-1,0],[-1,1],[0,1]])

        pos = np.argwhere(grid[:,:,1] == self.scale**1.0)[0]
        target = np.argwhere(grid[:,:,2] == self.scale*1.0)
        good_target = np.argwhere(grid[:,:,2] == self.real_scale*1.0)[0]
        new_pos = pos + act[action]

        # dist = math.sqrt((new_pos[0]-target[0])**2+(new_pos[1]-target[1])**2)
        dist_out = np.linalg.norm(new_pos - good_target)

        yaw = self.det_yaw(act[action])
        #reward = (dist1 - dist2)*(max_norm - dist2)
        #reward = -dist2
        car_grid = self.make_car_boound(car_grid,yaw,new_pos) # 현재 차량을 그린 car_grid 가져옴
        car_pos = np.where((car_grid[:,:,1]==255) & (car_grid[:,:,2]==255)) # car_grid로부터 차가 차지하는 좌표들 가져옴
        car_pos = list(zip(car_pos[0],car_pos[1]))
        reward += -1.0
        
        if (np.any(new_pos < 0.0) or new_pos[1] > (39.0)):
            #dist = np.linalg.norm(pos - target)
            #reward = (dist1 - dist2)
            return grid, reward, done, dist_out, car_grid, crack
        
        for car in car_pos :
            # 장애물 부딪히면 학습 종료 -> 우선적으로 판별함
            if new_grid[car[0],car[1],0] == 1.0:
                crack = True
                reward += -10.0
                return grid, reward, done, dist_out, car_grid, crack
        
        for car in car_pos :
            # 패딩 부분 밟으면 감점하고 이동
            if new_grid[car[0],car[1],0] == 255.0:
                reward += -2.0
                A = True
            
            # 차선 밟으면 감점하고 이동
            elif new_grid[car[0],car[1],0] == 2.0 : 
                reward += -5.0
                B = True

            if A or B:
                break
            
        new_grid[pos[0],pos[1],1] = 0.0
        new_grid[new_pos[0],new_pos[1],1] = self.scale*1.0
        
        # finish 조건 완화
        
        for car in car_pos:
            
            if ((car[0] == good_target[0]) and (car[1] == good_target[1])):
                # print("really good")
                reward = 200.0
                done_1 = True
                break
            
            elif any((car == t).all() for t in target):
                # print("good")
                reward = 200.0
                done_1 = True
                break

        return new_grid, reward, done_1, dist_out, car_grid, crack
    
    def get_tensor(self,grid):
        S = torch.Tensor(grid).transpose(2,1).transpose(1,0).unsqueeze(0)
        return S
    
    def render(self,grid):
        # imshow(grid)
        plot = imshow(grid)
        return plot
