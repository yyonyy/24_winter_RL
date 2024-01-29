#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pathlib
import rospy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from erp42_msgs.msg import DriveCmd, ModeCmd
from morai_msgs.msg import EgoVehicleStatus
from sensor_msgs.msg import Imu
from make_map_sim import Show
from tf.transformations import euler_from_quaternion
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

######변수 선언부#####
SCALE_CONST = 180
LAD = 200
################
WB = 1
K = 2.0
"""
조향각 -> 
    시계방향 : -
    반시계방향 : +
깎아줄 param ->
    SCALE_CONST : 시뮬상 환경과 opencv map scael 일치시키는 상수값    
    LAD : 목표를 넘기는 기준값. 클 수록 다음 목표를 더 빨리 바꿈 
"""

class Tracking_DQN :
    def __init__(self) :
        # rospy.init_node('Tracking_node', anonymous=False) 
        rospy.Subscriber('/Ego_topic',EgoVehicleStatus,self.odom_cb)
        rospy.Subscriber('/imu/data',Imu,self.imu_cb)
        self.cmd_pub = rospy.Publisher('/drive',DriveCmd, queue_size=10)
        self.image = np.zeros((1200,1200,3))
        self.map_make = Show()
        self.car_pose = []
        self.velocity_profile = 5
        self.v = 0
        self.idx = 0
        self.yaw = 0
        self.inf_yaw = 0
        self.start_x, self.start_y = (0,0)
        self.error_front_axle = 0
        self.head_first = True
        self.imu_first = True
        self.tracking_End = False
        self.scaled = False
        
    def rotate_point(self, want_x, want_y) : 
        '''
        좌표계 변환 함수
        input : MORAI coord
        output : start car coord 
        '''
        angle_rad = np.radians(self.heading-90) 
        rotation_matrix = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_point = np.dot(rotation_matrix, (want_x, want_y))
        return tuple(rotated_point.flatten())
        
    def imu_cb(self, data) :
        x = data.orientation.x
        y = data.orientation.y
        z = data.orientation.z
        w = data.orientation.w

        orientation_list = [x, y, z, w]
        if self.imu_first :
            _,_,self.yaw_first = euler_from_quaternion(orientation_list)
            self.yaw = self.yaw_first
            self.imu_first = False
        else :
            _,_,self.yaw = euler_from_quaternion(orientation_list) 
            self.yaw = self.yaw -self.yaw_first
        self.yaw += np.radians(90)
        self.yaw = self.normalize_angle(self.yaw)

    def odom_cb(self, data) :
        '''
        tracking 과정 간단히 설명 :
            1. 시작점기준으로 odometry, map 생성
            2. 현재의 odometry를 받아오고, 시작점과의 차이를 바탕으로 OpenCV좌표계 상의 odometry로 변경
            3. 위 값들을 바탕으로 차량의 현재 위치 추정 (base_link를 LiDAR랑 맞춰줘야 할 듯 이미 맞는건가..?)
        '''
        self.x = data.position.x
        self.y = data.position.y
        self.v = data.velocity.x
        
        if self.head_first :
            self.heading = data.heading
            self.head_first = False
            self.start_x = data.position.x
            self.start_y = data.position.y

        moved_x = self.x - self.start_x
        moved_y = self.y - self.start_y
        CV_start = np.array([600, 1199]) 
        cv_odom = np.array(self.rotate_point(moved_x, moved_y))*SCALE_CONST
        # print(cv_odom)
        cv_odom[1] = -cv_odom[1] # 전진 -> openCV 기준 -y 방향
        self.car_pose = (CV_start + cv_odom).astype(int)
        cv2.circle(self.image, self.car_pose, 3, (255,255,255), -1)
        # cv2.imshow('imshow',self.image)
        # print('car pose : ',self.car_pose)
    
    def normalize_angle(self,angle):

        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle    
        
         
    def scaling(self, not_scaled_x, not_scaled_y) :
        '''
        최초 path 1회 가져올때만 실행
        기존의 dqn path를 1200x1200으로 변경  
        '''
        cnt = 0
        H,W = 40, 40
        path_x, path_y = [], []
        for i in range(len(not_scaled_x)) :
            if cnt%4 == 0 :
                x = int((not_scaled_x[i]/W)*1200)
                # if x>599 : x=599
                y = int((not_scaled_y[i]/H)*1200)
                # if y>599 : y=599
                path_x.append(x+30)
                path_y.append(y)
                cv2.circle(self.image,(x+30,y),LAD,[0,0,255],1)
                cv2.circle(self.image,(x+30,y),1,[0,255,0],-1)
            cnt+=1 
        self.scaled = True
        return path_x, path_y
                        
    def select_target(self, curr_idx, path_x, path_y) :
        distance = math.sqrt((self.car_pose[0]-path_x[curr_idx+1])**2 + (self.car_pose[1]-path_y[curr_idx+1])**2)
        
        # print("dis: ",distance)
        # print("x",path_x)
        # print("y",path_y)
        
        if distance <= LAD :
            curr_idx +=1 
        return curr_idx
    
    def calc_error(self, path_x, path_y) :
        fx = self.car_pose[0] + WB * np.cos(self.yaw)
        fy = self.car_pose[1] - WB * np.sin(self.yaw)
        
        front_axle_vec = [np.cos(self.yaw + np.pi / 2),
                        np.sin(self.yaw + np.pi / 2)]
        
        dx = path_x[self.idx] - fx
        dy = path_y[self.idx] - fy
        
        error_front_axle = np.dot([dx,-dy], front_axle_vec)
        if self.idx >= len(path_x)-2 :
            x = path_x[self.idx] - path_x[self.idx-1]
            y = -(path_y[self.idx] - path_y[self.idx-1])
        else :
            x = path_x[self.idx+1] - path_x[self.idx]
            y = -(path_y[self.idx+1] - path_y[self.idx])
        inf_yaw = self.normalize_angle(math.atan2(y,x))
        # print('======= inf_yaw : ', inf_yaw)
        # print('========== error_front_axle ', error_front_axle)
        
        return inf_yaw, error_front_axle

    def stanley(self, path_x, path_y) :
        '''
        대충 횡방향 에러, 헤딩 에러 구하고 delta 뽑기  
        ''' 
        if self.car_pose == []:
            DriveCmd_msg = DriveCmd()
            DriveCmd_msg.KPH = self.velocity_profile
            DriveCmd_msg.Deg = int(0) 
            
            ModeCmd_msg = ModeCmd()
            ModeCmd_msg.MorA = 0x01
            ModeCmd_msg.Gear = 0x00
            ModeCmd_msg.EStop = 0x00
            return ModeCmd_msg, DriveCmd_msg
        
        path_x 
        
        # 일단 목표점 가능하면 갱신
        if self.idx <= len(path_x)-2 :
            # print("idx, len(path) : ", self.idx, len(path_x))
            self.idx = self.select_target(self.idx, path_x, path_y)
            self.inf_yaw, self.error_front_axle = self.calc_error(path_x,path_y)
        elif self.idx > len(path_x)-2 : # 마지막 index 예외처리
            dist = math.sqrt((self.car_pose[0]-path_x[self.idx])**2 + 
                             (self.car_pose[1]-path_y[self.idx])**2)
            if dist <= 70  :
                self.tracking_End = True
                # self.head_first = True
                # self.imu_first = True
                # self.scaled = False
                self.image = np.zeros((1200,1200,3))
                self.idx =0
                
            self.inf_yaw, self.error_front_axle = self.calc_error(path_x,path_y)
        # theta_e corrects the heading error
        theta_e = self.normalize_angle(self.inf_yaw - self.yaw)
        # theta_d corrects the cross track error
        theta_d = self.normalize_angle(np.arctan2(K * self.error_front_axle, self.v))
        # Steering control
        delta = theta_e + theta_d
        delta = delta * (180/np.pi) / 10

        # print('===========inf_yaw, curr_yaw : ',self.inf_yaw, self.yaw)
        # print('==============theta_e : ',theta_e)
        # print('===========theta_d : ',theta_d)
        # print('===delta : ', delta)
        DriveCmd_msg = DriveCmd()
        DriveCmd_msg.KPH = self.velocity_profile
        DriveCmd_msg.Deg = int(delta) 
        # print('curridx, max : ',self.idx, len(path_x)-1)
        ModeCmd_msg = ModeCmd()
        ModeCmd_msg.MorA = 0x01
        ModeCmd_msg.Gear = 0x00
        ModeCmd_msg.EStop = 0x00
        
        if self.car_pose[1] <= -30 : # 비상 종료 조건
            self.tracking_End = True
            self.image = np.zeros((1200,1200,3))
            self.idx =0
        
        return ModeCmd_msg, DriveCmd_msg
        
    # def puuuuub(self) :
        # DriveCmd_msg = DriveCmd()
        # ModeCmd_msg = ModeCmd()

        # DriveCmd_msg.KPH = self.velocity_profile 
        # DriveCmd_msg.Deg = int(self.stanley()) 
        # ModeCmd_msg.MorA = 0x01
        # ModeCmd_msg.Gear = 0x00
        # ModeCmd_msg.EStop = 0x00

        # self.cmd_pub.publish(DriveCmd_msg) 
        