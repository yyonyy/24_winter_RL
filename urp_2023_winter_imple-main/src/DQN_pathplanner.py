import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Models import ConvNet
from copy import deepcopy as dc

TRY_TIMES = 100

class Planning :
    def __init__(self) -> None:
        self.i = 0 
        self.Nobs = 2
        self.Dobs = 2       
        # 차 전용이라서 act dim 3 줍니다. 
        self.model = ConvNet(40,40,3,3).cuda() 
        self.done = False
        
    def planning(self, map) :
        env = map
        self.model.load_state_dict(torch.load("/home/heven/heven_ws/src/push_plz/model/model_bb.pt"))
        image_mean = torch.load("/home/heven/heven_ws/src/push_plz/model/norm_bb.pt")
        
        cum_obs = dc(env)
        obs = dc(env)
        trajectory = [obs]
        done = False
        state = torch.Tensor(obs).transpose(2,1).transpose(1,0).unsqueeze(0)
        sum_r = 0
        epsilon = 0.0 
        
        coordinate_list = []
        coordinate_x = []
        coordinate_y = []

        for t in range(TRY_TIMES): 
            Q = self.model(state.cuda() - image_mean)
            num = np.random.rand()
            if (num < epsilon):
                action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
            else:
                action = torch.argmax(Q,dim=1)
            new_obs, reward, done, dist = self.step(obs,action.item())

            trajectory.append(new_obs)
            new_state = torch.Tensor(new_obs).transpose(2,1).transpose(1,0).unsqueeze(0)
            sum_r = sum_r + reward
            state = dc(new_state)
            obs = dc(new_obs)
            cum_obs[:,:,1] += obs[:,:,1]
            if done:
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
        coordinate_x = np.array(coordinate_x)
        coordinate_y = np.array(coordinate_y)    
        # print("main:::",coordinate_x,coordinate_y)
        # 5차 다항식으로 보간
        # coefficients = np.polyfit(coordinate_x, coordinate_y, 1)
        # poly = np.poly1d(coefficients)

        # # 보간 결과를 평가할 x 범위 설정
        # x_interpolated = np.linspace(min(coordinate_x), max(coordinate_x), 100)

        # # 다항식을 통한 보간 결과 계산
        # y_interpolated = poly(x_interpolated)
        
        # num_points = 10
        # x_representative = np.linspace(min(coordinate_x), max(coordinate_x), num_points)
        # y_representative = poly(x_representative)
        
        # x,y = self.scaling_path(x_representative, y_representative)
        
        return coordinate_x, coordinate_y, trajectory
        # return x_representative, y_representative, trajectory
                
    def step(self,grid,action):
        # max_norm = self.N
        new_grid = dc(grid)
        done = False
        reward = -0.5
        # act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        act = np.array([[0,1],[-1,0],[0,-1]])
        
        pos = np.argwhere(grid[:,:,1] == 10.0)[0]
        target = np.argwhere(grid[:,:,2] == 10.0)
        new_pos = pos + act[action]
        dist1 = np.linalg.norm(pos - target)
        dist2 = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)*(max_norm - dist2)
        #reward = -dist2
        # reward = -1.0

        if (np.any(new_pos < 0.0) or new_pos[1] > (40 - 1) or new_pos[0] > (40 -1)):
            #dist = np.linalg.norm(pos - target)
            #reward = (dist1 - dist2)
            reward += -5.0
            return grid, reward, done, dist2
        
        # if (grid[new_pos[0],new_pos[1],0] == 1.0):
        #     return grid, reward, done, dist2
        
        # surplus
        if (grid[new_pos[0],new_pos[1],0] == 255.0):
            reward += -0.7
        
        # obs
        elif (grid[new_pos[0],new_pos[1],0] == 1.0):
            reward += -2.0
            return grid, reward, done, dist2
        
        new_grid[pos[0],pos[1],1] = 0.0
        new_grid[new_pos[0],new_pos[1],1] = 10.0
        
        # if ((new_pos[0] == target[0]) and (new_pos[1] == target[1])):
        #     reward += 100.0
        #     done = True
        
        if any((new_pos == t).all() for t in target):
            reward = 100.0
            done = True
            self.done = True
        #dist = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)
        return new_grid, reward, done, dist2
        
        
    def visualize_episode(self, trajectory):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        # img[:, :, 3] = 255
        
        img[trajectory[0][:, :, 0] == 1.0] = [255, 0, 0]  #장애물

        img[trajectory[0][:, :, 0] == 255.0] = [255, 0, 255]  #surplus

        for obs in trajectory:
            pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
            img[pos[0], pos[1]] = [255, 255, 0]  #이동 경로

        initial = np.argwhere(trajectory[0][:, :, 1] == 10.0)[0]
        img[initial[0], initial[1]] = [0, 255, 0]  #시작 위치
        
        target = np.argwhere(trajectory[0][:, :, 2] == 10.0)
        for t in target:
            # print(t)
            img[t[0],t[1]] = [0,0,255]
        
        plt.imshow(img)
        plt.show()
        plt.pause(1)
        
        # return img