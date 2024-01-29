"""
@author: orrivlin
"""

import torch 
import numpy as np
import copy
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from collections import deque
from Models import ConvNet, ConvNet_noPool
from log_utils import logger, mean_val
from HER import HER
from copy import deepcopy as dc


class DQN_HER:
    def __init__(self, env, gamma, buffer_size, ddqn):
        self.env = env
        [Sdim,Adim] = env.get_dims()
        # pooling
        self.model = ConvNet(Sdim[0],Sdim[1],3,Adim).cuda()
        # no pooling
        self.model = ConvNet_noPool(Sdim[0],Sdim[1],3,Adim).cuda()
        self.target_model = copy.deepcopy(self.model).cuda()
        self.her = HER()
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.batch_size = 16
        self.epsilon = 0.1
        self.buffer_size = buffer_size
        self.step_counter = 0
        self.epsi_high = 0.9
        self.epsi_low = 0.1
        self.steps = 0
        self.count = 0
        self.decay = 2000
        self.eps = self.epsi_high
        self.update_target_step = 3000
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('avg_loss')
        self.log.add_log('final_dist')
        self.log.add_log('buffer')
        self.image_mean = 0
        self.image_std = 0
        self.ddqn = ddqn
        
        self.previous_action = 2
        self.replay_buffer = deque(maxlen=buffer_size)
        
    def run_episode(self, i):
        self.her.reset()
        ########
        obs, done = self.env.reset()
        ########
        done_1 = False
        done = False
        state = self.env.get_tensor(obs)
        sum_r = 0
        mean_loss = mean_val()
        min_dist = 100000
        max_t = 100
        previous_action = self.previous_action
        
        ############################################
        trajectory = [obs]
        trajectory2 = [obs]
        ############################################

        for t in range(max_t):
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * self.steps/self.decay))
            Q = self.model(self.norm(state.cuda()))
            num = np.random.rand()

            if (num < self.eps):
                action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
                # possible_actions = []
                # if previous_action == 0:
                #     possible_actions = [0, 1]
                # elif previous_action == 1:
                #     possible_actions = [0, 1, 2]
                # elif previous_action == 2:
                #     possible_actions = [1, 2, 3]
                # elif previous_action == 3:
                #     possible_actions = [2, 3, 4]
                # elif previous_action == 4:
                #     possible_actions = [3, 4]

                # action = np.random.choice(possible_actions)
                # action = torch.LongTensor([action])

            else:
                action = torch.argmax(Q,dim=1)
            ############################################
            
            # print("previous :",previous_action,"current :",action)

            new_obs, reward, done_1, dist, car_grid, crack = self.env.step(obs,action.item(),previous_action)
            previous_action = action.item()

            if not crack:
                Print = True
                trajectory2.append(car_grid)
            new_state = self.env.get_tensor(new_obs)
            sum_r = sum_r + reward
            if dist < min_dist:
                min_dist = dist
            if (t+1) == max_t:
                done = True
            
            self.replay_buffer.append([dc(state.squeeze(0).numpy()),dc(action),dc(reward),dc(new_state.squeeze(0).numpy()),dc(done)])
            self.her.keep([state.squeeze(0).numpy(),action,reward,new_state.squeeze(0).numpy(),done])
            loss = self.update_model()
            mean_loss.append(loss)
            state = dc(new_state)
            obs = dc(new_obs)
            
            ############################################
            trajectory.append(new_obs)
            ############################################
            
            self.step_counter = self.step_counter + 1
            if (self.step_counter > self.update_target_step):
                self.target_model.load_state_dict(self.model.state_dict())
                self.step_counter = 0
                print('updated target model')
            if done or done_1: 
                break
                
                
        ##################################
        if i % 20 ==0:
            self.visualize_episode(trajectory, trajectory2)
        ##################################

        print("!!!!!!!!!!!!!!1reward :",sum_r)
        
        her_list = self.her.backward()
        for item in her_list:
            self.replay_buffer.append(item)
        
        # if min_dist <= 3.0:
        #     if min_dist == 0.0:
        #         print("really good!")
        #     else:
        #         print("good!")
        #     min_dist = 0.0    
        if done_1:
            print("good!")
            min_dist = 0.0
        
        self.log.add_item('tot_return',sum_r)
        self.log.add_item('avg_loss',mean_loss.get())
        self.log.add_item('final_dist',min_dist)
        
    def gather_data(self):
        self.her.reset()
        obs, done = self.env.reset()
        done = False
        done_1 = False
        state = self.env.get_tensor(obs)
        sum_r = 0
        min_dist = 100000
        max_t = 100
        previous_action = self.previous_action

        for t in range(max_t):
            self.eps = 1.0
            Q = self.model(state.cuda())
            num = np.random.rand()
            if (num < self.eps):
                action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
                # possible_actions = []
                # if previous_action == 0:
                #     possible_actions = [0, 1]
                # elif previous_action == 1:
                #     possible_actions = [0, 1, 2]
                # elif previous_action == 2:
                #     possible_actions = [1, 2, 3]
                # elif previous_action == 3:
                #     possible_actions = [2, 3, 4]
                # elif previous_action == 4:
                #     possible_actions = [3, 4]

                # action = np.random.choice(possible_actions)
                # action = torch.LongTensor([action])
            else:
                action = torch.argmax(Q,dim=1)
            new_obs, reward, done_1, dist, _, _ = self.env.step(obs,action.item(),previous_action)
            previous_action = action.item()

            new_state = self.env.get_tensor(new_obs)
            sum_r = sum_r + reward
            if dist < min_dist:
                min_dist = dist
            if (t+1) == max_t:
                done = True
            
            self.replay_buffer.append([dc(state.squeeze(0).numpy()),dc(action),dc(reward),dc(new_state.squeeze(0).numpy()),dc(done)])
            state = dc(new_state)
            obs = dc(new_obs)
        return min_dist

    def calc_norm(self):
        S0, A0, R1, S1, D1 = zip(*self.replay_buffer)
        S0 = torch.tensor( S0, dtype=torch.float)
        self.image_mean = S0.mean(dim=0).cuda()
        self.image_std = S0.std(dim=0).cuda()
        
    def norm(self,state):
        return state - self.image_mean
        
    def update_model(self):
        self.optimizer.zero_grad()
        num = len(self.replay_buffer)
        K = np.min([num,self.batch_size])
        samples = random.sample(self.replay_buffer, K)
        
        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.tensor( S0, dtype=torch.float)
        A0 = torch.tensor( A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor( R1, dtype=torch.float).view(K, -1)
        S1 = torch.tensor( S1, dtype=torch.float)
        D1 = torch.tensor( D1, dtype=torch.float)
        
        S0 = self.norm(S0.cuda())
        S1 = self.norm(S1.cuda())
        if self.ddqn == True:
            model_next_acts = self.model(S1).detach().max(dim=1)[1]
            target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1).gather(1,model_next_acts.unsqueeze(1)).squeeze()*(1 - D1.cuda())
        else:
            target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1).max(dim=1)[0].detach()*(1 - D1.cuda())
        policy_q = self.model(S0).gather(1,A0.cuda())
        L = F.smooth_l1_loss(policy_q.squeeze(),target_q.squeeze())
        L.backward()
        self.optimizer.step()
        return L.detach().item()
    
    def run_epoch(self, i):
        self.run_episode(i)
        self.log.add_item('buffer',len(self.replay_buffer))
        return self.log


    # ##################################
    # plt.ion()
    # def visualize_episode(self, trajectory):
    #     img = np.zeros((20, 40, 3), dtype=np.uint8)
    #     # img[:, :, 3] = 255
        
    #     img[trajectory[0][:, :, 0] == 1.0] = [255, 0, 0]  #장애물

    #     img[trajectory[0][:, :, 0] == 255.0] = [255, 0, 255]  #surplus

    #     for obs in trajectory:
    #         pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
    #         img[pos[0], pos[1]] = [255, 255, 0]  #이동 경로
        
    #     initial = np.argwhere(trajectory[0][:, :, 1] == self.env.scale)[0]
    #     img[initial[0], initial[1]] = [0, 255, 0]  #시작 위치
        
    #     target = np.argwhere(trajectory[0][:, :, 2] == self.env.scale)[0]
    #     img[target[0], target[1]] = [0, 0, 255]  #목표 위치

        # plt.imshow(img)
        # plt.pause(0.1)
    # plt.ioff()
    
    plt.ion()
    def visualize_episode(self,trajectory_1, trajectory_2):
        # _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 첫번째 모델 Visualization
        img_1 = np.zeros((40, 40, 3), dtype=np.uint8)
        img_1[trajectory_1[0][:, :, 0] == 1.0] = [255, 0, 0]  # 장애물 red 
        img_1[trajectory_1[0][:, :, 0] == 2.0] = [255, 255, 255]  # 차선 white
        img_1[trajectory_1[0][:, :, 0] == 255.0] = [255, 0, 255]  #surplus pink

        for obs in trajectory_1:
            pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
            img_1[pos[0], pos[1]] = [255, 255, 0]  # 이동 경로
        img_1[pos[0], pos[1]] = [0, 255, 255]

        initial = np.argwhere(trajectory_1[0][:, :, 1] == self.env.scale)[0]
        img_1[initial[0], initial[1]] = [0, 255, 0]  # 시작 위치

        target = np.argwhere(trajectory_1[0][:, :, 2] == self.env.scale)
        for t in target:
            img_1[t[0],t[1]] = [0,200,200]

        target_1 = np.argwhere(trajectory_1[0][:, :, 2] == self.env.real_scale)[0]
        img_1[target_1[0], target_1[1]] = [0, 0, 255]  # 목표 위치

        plt.subplot(1, 2, 1)
        plt.imshow(img_1)
        plt.title('Model 1')

        # 차가 지나간 자리 visualization
        img_2 = np.zeros((40, 40, 3), dtype=np.uint8)
        img_2[trajectory_2[0][:, :, 0] == 1.0] = [255, 0, 0]  # 장애물 red
        img_2[trajectory_2[0][:, :, 0] == 2.0] = [255, 255, 255] # 차선 white
        img_2[trajectory_2[0][:, :, 0] == 255.0] = [255, 0, 255]  #surplus pink
        
        for cars in trajectory_2 :
            car_pos = np.where((cars[:,:,1]==255) & (cars[:,:,2]==255))
            car_pos = list(zip(car_pos[0],car_pos[1]))

            for car in car_pos :
                img_2[car[0], car[1]] = [0,255,255]
            img_2[pos[0], pos[1]] = [255, 0, 255] # 마지막위치
            
        initial = np.argwhere(trajectory_2[0][:, :, 1] == self.env.scale)[0]
        img_2[initial[0], initial[1]] = [0, 255, 0]  # 시작 위치
            
        target = np.argwhere(trajectory_1[0][:, :, 2] == self.env.scale)
        for t in target:
            img_2[t[0],t[1]] = [0,200,200]

        target_1 = np.argwhere(trajectory_1[0][:, :, 2] == self.env.real_scale)[0]
        img_2[target_1[0], target_1[1]] = [0, 0, 255]  # 목표 위치
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_2)
        plt.title('Model 2')

        plt.draw()
        plt.pause(0.1)
    plt.ioff()