#!/usr/bin/env python3
import sys, os
from time import time

import rospy
import numpy as np
from math import *
from collections import deque
from lane_control.msg import lane_info # int16 left_x, int16 right_x, float32 left_theta, float32 right_theta
from erp42_msgs.msg import ModeCmd
from erp42_msgs.msg import DriveCmd

class Lane_Following_Control():
    def __init__(self):
        self.lane_width = 3
        # rospy.init_node('lfc_node', anonymous=False)
        rospy.Subscriber('/lane_result', lane_info, self.lane_cb)
        self.cmd_pub = rospy.Publisher('/drive', DriveCmd, queue_size=10)

        self.lat_err_filtered = 0
        self.head_err_filtered = 0

    def lane_cb(self, data):
        self.left = data.left_x
        self.right = data.right_x
        self.left_theta = radians(data.left_theta)
        self.right_theta = radians(data.right_theta)

    def moving_filter(self, input, horizon=5, weights = [55/85, 20/85, 7/85, 3/85, 1/85]): # [5/15, 4/15, 3/15, 2/15, 1/15], [55/85, 20/85, 7/85, 3/85, 1/85]
        queue = deque([0 for _ in range(horizon)], maxlen=horizon)
        queue.append(input)
        result = [a*b for a,b in zip(queue, weights)]
        return sum(result)

    def lfc_controller(self):
        ############ Control Algorithm ############
        ########## your code starts here ##########
        if self.left == 160:
            lateral_err = (0.5 - (self.right/320))*self.lane_width
            heading_err = pi/2 - self.right_theta
        elif self.right == 160:
            lateral_err = (-0.5 + (self.left/320))*self.lane_width
            heading_err = -pi/2 + self.left_theta
        else:  
            lateral_err = ((self.left/(self.left + self.right)) - 0.5)*self.lane_width # right(+) left (-)
            heading_err = pi/2 - ((self.left_theta + self.right_theta)/2) # right(+) left (-)

        self.lat_err_filtered = self.moving_filter(lateral_err)
        self.head_err_filtered = self.moving_filter(heading_err)

        # rospy.loginfo('lateral error : {}'.format(self.lat_err_filtered))
        # rospy.loginfo('heading error : {}'.format(degrees(self.head_err_filtered)))
        #  
        ## getting the curvature of a given looahead horizon would worth consideration ##
        ## configure the `velocity` profile w.r.t curvature ##
        velocity_profile = 7 # velocity profile is set to constant for now

        k = 100
        k_s = 0

        stanley_cmd = self.head_err_filtered + atan(k*self.lat_err_filtered/((velocity_profile/3.6) + k_s))     
        # 최대 조향각, 최소 조향각을 고려해서 노멀라이즈
        stanley_cmd = (28*pi/180) * stanley_cmd/(2*pi/3) * (180/pi)
        # rospy.loginfo('stanley : {}'.format(stanley_cmd))
        ########### your code ends here ###########

        DriveCmd_msg = DriveCmd()
        DriveCmd_msg.KPH = velocity_profile 
        DriveCmd_msg.Deg = int(stanley_cmd) 
        
        ModeCmd_msg = ModeCmd()
        ModeCmd_msg.MorA = 0x01
        ModeCmd_msg.Gear = 0x00
        ModeCmd_msg.EStop = 0x00
        
        return ModeCmd_msg, DriveCmd_msg 