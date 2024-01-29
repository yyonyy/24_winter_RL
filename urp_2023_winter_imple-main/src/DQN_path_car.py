import torch
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Models import ConvNet
from copy import deepcopy as dc

TRY_TIMES = 100

class Planning :
    def __init__(self) -> None:
        self.i = 0   
        # 차 전용이라서 act dim 5 줍니다.
        self.model = ConvNet(40,40,3,5).cuda() 
        self.done = False
        self.obs = np.zeros((40,40,3))
        self.state = torch.Tensor(self.obs).transpose(2,1).transpose(1,0).unsqueeze(0)
        self.model.load_state_dict(torch.load("/home/heven/heven_ws/src/push_plz/model/model.pt"))
        self.image_mean = torch.load("/home/heven/heven_ws/src/push_plz/model/norm.pt")
        self.Q = self.model(self.state.cuda() - self.image_mean)
    
    # 2차 방정식으로 근사
    def quadratic_function(self, x, a, b, c):
        return a * x ** 2 + b * x + c
    
    def planning(self, map) :
        env = map
        cum_obs = dc(env)
        self.obs = dc(env)
        self.state = torch.Tensor(self.obs).transpose(2,1).transpose(1,0).unsqueeze(0)
        
        trajectory = [self.obs]
        done_1 = False
        sum_r = 0

        coordinate_list = []
        coordinate_x = []
        coordinate_y = []

        for t in range(TRY_TIMES): 
            self.Q = self.model(self.state.cuda() - self.image_mean)
            action = torch.argmax(self.Q,dim=1)
            new_obs, reward, self.done = self.step(self.obs,action.item())

            trajectory.append(new_obs)
            new_state = torch.Tensor(new_obs).transpose(2,1).transpose(1,0).unsqueeze(0)
            sum_r = sum_r + reward
            self.state = dc(new_state)
            self.obs = dc(new_obs)
            cum_obs[:,:,1] += self.obs[:,:,1]
            if self.done:
                print('Success!!!')
                # success_num += 1
                break
            elif t == TRY_TIMES-1 :
                print('fail...')
                
        for obs in trajectory:
            pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
            coordinate = [pos[0],pos[1]]
            coordinate_list.append(coordinate)
            coordinate_y.append(pos[0])
            coordinate_x.append(pos[1]) 
        
        coordinate_list = np.array(coordinate_list)
        
        ###########그냥 하기
        # x_representative = np.array(coordinate_x)
        # y_representative = np.array(coordinate_y) 
        ##########################################
        
        # coordinate_x = np.array(coordinate_x)
        # coordinate_y = np.array(coordinate_y)    
        # print("origin:::",coordinate_x,coordinate_y)
        # # 5차 다항식으로 보간
        # coefficients = np.polyfit(coordinate_x, coordinate_y, 1)
        # poly = np.poly1d(coefficients)

        # # 보간 결과를 평가할 x 범위 설정
        # x_interpolated = np.linspace(min(coordinate_x), max(coordinate_x), 100)

        # # # 다항식을 통한 보간 결과 계산
        # y_interpolated = poly(x_interpolated)
        
        # num_points = 30
        # x_representative = np.linspace(min(coordinate_x), max(coordinate_x), num_points)
        # y_representative = poly(x_representative)
        
        # # x,y = self.scaling_path(x_representative, y_representative)
        # print("represen:::",x_representative,y_representative)
        # # return coordinate_x, coordinate_y, trajectory
        
        
        #######################################################################################
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img_1 = np.zeros((40, 40, 3), dtype=np.uint8)
        
        for point in coordinate_list:
            img[point[0], point[1]] = [255, 255, 0]  #이동 경로
        img = np.rot90(img, k=-1)

        zero_coordinates = np.argwhere(~np.all(img == [0, 0, 0], axis=-1))

        x_coor = zero_coordinates[:, 1]
        y_coor = zero_coordinates[:, 0]
        
        # curve_fit 하는 부분
        popt, pcov = curve_fit(self.quadratic_function, x_coor, y_coor)
        # popt, pcov = curve_fit(cubic_function, x_coordinates, y_coordinates)

        a_opt, b_opt, c_opt = popt
        # a_opt, b_opt, c_opt,d_opt = popt

        x_fit = np.linspace(min(x_coor), max(x_coor), 100)
        y_fit = self.quadratic_function(x_fit, a_opt, b_opt, c_opt)
        
        points_1 = np.column_stack((x_fit,y_fit))
        
        for point_1 in points_1:
            img_1[int(point_1[1]), int(point_1[0])] = [255,255,255]
        
        img = np.rot90(img, k=1)
        img_1 = np.rot90(img_1, k=1)
        
        # cv2.imshow("2",img)
        # cv2.imshow("1",img_1)
        # cv2.waitKey(0)
        
        represenatative_coordinates = np.argwhere(~np.all(img_1 == [0, 0, 0], axis=-1))

        x_representative = represenatative_coordinates[:, 1]
        y_representative = represenatative_coordinates[:, 0]
        
        return x_representative, y_representative, trajectory
        
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
    
             
    def step(self,grid,action):
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

        # pos = np.argwhere(grid[:,:,1] == 10)[0]
        # target = np.argwhere(grid[:,:,2] == 10)
        # good_target = np.argwhere(grid[:,:,2] == self.real_scale*1.0)[0]
        # new_pos = pos + act[action]
        
        pos = np.argwhere(grid[:,:,1] == 10.0)[0]
        target = np.argwhere(grid[:,:,2] == 10.0)

        new_pos = pos + act[action]
        good_target = np.argwhere(grid[:,:,2] == 20.0)[0]
        
        # dist = math.sqrt((new_pos[0]-target[0])**2+(new_pos[1]-target[1])**2)
        # dist_out = np.linalg.norm(new_pos - good_target)

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
            return grid, reward, done
        
        for car in car_pos :
            # 장애물 부딪히면 학습 종료 -> 우선적으로 판별함
            if new_grid[car[0],car[1],0] == 1.0:
                crack = True
                reward += -10.0
                return grid, reward, done
        
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
        new_grid[new_pos[0],new_pos[1],1] = 10*1.0
        
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
        
        # if any((new_pos == t).all() for t in target):
        #     reward = 100.0
        #     done = True
        

        return new_grid, reward, done_1
        
        
    def visualize_episode(self, trajectory):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        # img[:, :, 3] = 255
        
        img[trajectory[0][:, :, 0] == 1.0] = [255, 0, 0]  #장애물

        img[trajectory[0][:, :, 0] == 255.0] = [255, 0, 255]  #surplus
        
        img[trajectory[0][:, :, 0] == 2.0] == [255,255,255] #lane

        for obs in trajectory:
            pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
            img[pos[0], pos[1]] = [255, 255, 0]  #이동 경로

        initial = np.argwhere(trajectory[0][:, :, 1] == 10.0)[0]
        img[initial[0], initial[1]] = [0, 255, 0]  #시작 위치
        
        target = np.argwhere(trajectory[0][:, :, 2] == 10.0)
        for t in target:
            # print(t)
            img[t[0],t[1]] = [0,0,255]
        target_1 = np.argwhere(trajectory[0][:, :, 2] == 20)[0]
        img[target_1[0], target_1[1]] = [0, 0, 255]  # 목표 위치
        
        plt.imshow(img)
        plt.show()
        plt.pause(1)
        

        
        # return img