# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 07:57:53 2019

@author: orrivlin
"""

import cv2
import torch 
import numpy as np
import random
import matplotlib.pyplot as plt
from Models import ConvNet
from Nav2D_Yeong import Navigate2D
from copy import deepcopy as dc

TRY_TIMES = 100
success_num = 0
epochs = 100
Nobs = 1
Dobs = 2
Slope = random.randint(30,90)
env = Navigate2D(Nobs,Dobs,Slope)
[Sdim,Adim] = env.get_dims()
model = ConvNet(Sdim[0],Sdim[1],3,Adim).cuda()
model.load_state_dict(torch.load('./pt_files/model_best.pt'))
image_mean = torch.load('./pt_files/norm_best.pt').cuda()

def visualize_episode(trajectory):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img_1 = np.zeros((40, 40, 3), dtype=np.uint8)
    img[trajectory[0][:, :, 0] == 1.0] = [255, 0, 0]  #장애물

    for obs in trajectory:
        pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
        img[pos[0], pos[1]] = [255, 255, 0]  #이동 경로
        img_1[pos[0], pos[1]] = [255, 255, 0]
    
    initial = np.argwhere(trajectory[0][:, :, 1] == env.scale)[0]
    img[initial[0], initial[1]] = [0, 255, 0]  #시작 위치
    
    target = np.argwhere(trajectory[0][:, :, 2] == env.scale)[0]
    img[target[0], target[1]] = [0, 0, 255]  #목표 위치

    img_1 = cv2.resize(img, dsize=(400, 200), interpolation=cv2.INTER_LINEAR)

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
    plt.imshow(img)
    plt.title('Image')

    # # Display img_1
    # plt.subplot(1, 2, 2)  # 1 row, 2 columns, position 2
    # plt.imshow(img_1)
    # plt.title('Image_1')

    plt.show()
    plt.pause(2)

for i in range(epochs) :
    start_obs, done = env.reset()
    cum_obs = dc(start_obs)
    obs = dc(start_obs)
    trajectory = [obs]
    done = False
    state = env.get_tensor(obs)
    sum_r = 0
    epsilon = 0.0

    for t in range(TRY_TIMES): 
        Q = model(state.cuda() - image_mean)
        num = np.random.rand()
        if (num < epsilon):
            action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
        else:
            action = torch.argmax(Q,dim=1)
        new_obs, reward, done, _,_,_ = env.step(obs,action.item(),1)
        # 현재 포즈와 골 포즈를 프린트 하는 아래 코드 두 줄 추가
        # print("now pose : ", np.argwhere(new_obs[:,:,1] == 10.0)[0], end=' ')
        # print("goal pose : ",np.argwhere(new_obs[:,:,2] == 10.0)[0])

        trajectory.append(new_obs)
        new_state = env.get_tensor(new_obs)
        sum_r = sum_r + reward
        state = dc(new_state)
        obs = dc(new_obs)
        cum_obs[:,:,1] += obs[:,:,1]
        if done:
            break

    plt.ion()
    visualize_episode(trajectory)
    plt.ioff()
                    
    # env.render(cum_obs)
    print('time: {}'.format(t))
    print('return: {}'.format(sum_r))
