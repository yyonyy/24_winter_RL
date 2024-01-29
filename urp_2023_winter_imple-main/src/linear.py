#!/usr/bin/env python3

import rospy
import cv2
import numpy as np 
import matplotlib.pyplot as plt

from lane_follower import Lane_Following_Control
from tracking import Tracking_DQN
from DQN_pathplanner import Planning
from make_map_sim import Show

from erp42_msgs.msg import ModeCmd, DriveCmd
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class URP :
    def __init__(self) -> None:
        rospy.init_node("URP_DQN_node")
        self.lane_tracker = Lane_Following_Control()
        self.planner = Planning()
        
        self.map_make = Show()
        
        self.go_tracking = False
        
        self.map = np.zeros((40,40,3))
        
        self.trigger = False
        # self.trigger_sub = rospy.Subscriber("/tracking", String, self.trig_callback)
        
        self.mode_pub = rospy.Publisher("/mode", ModeCmd, queue_size=10)
        self.drive_pub = rospy.Publisher("/drive", DriveCmd, queue_size=10)

        self.modecmd_msg = ModeCmd()
        self.drivecmd_msg = DriveCmd()
        self.rate = rospy.Rate(10)
        
        while not rospy.is_shutdown() :
            self.main()
            self.rate.sleep()
                
    # def trig_callback(self, data) :
    #     # 트리거 펍주는 노드 하나 만들어야 함. 
    #     self.trigger = data.data
    
    def main(self) :
        """
        Pipe_Line : 
            1. Lane tracking 
            2. 장애물 트리거 켜지면 subscribe받은 map에서 planning 진행
            3. planning path 받은 후에 tracking  
            4. tracking 종료 되면 trigger, False로 초기화 후 다시 1로 돌아가서 반복반복
        """
        
        """
        Trigger 관련 내용
        make_map_sim 242번째 줄에서 일정 거리 이하면 -> trig =True
        314번째 줄에서 combined image 만들어지고 나면 다시 trig = False
        계속 반복
        """
        
        self.trigger = self.map_make.trig
        self.map = self.map_make.combined_image
        
        if not self.trigger :
            rospy.loginfo("====Lane Tracking... ")
            self.modecmd_msg, self.drivecmd_msg = self.lane_tracker.lfc_controller()
            # a = self.map_make.current_frame
            # cv2.imshow("1",a)
            # cv2.waitKey(1)
            
        elif self.trigger :
            # 최초 1회만 planning 진행
            if not self.go_tracking :
                rospy.loginfo("==============Planning !!")
                # self.path_x, self.path_y, cw = self.planner.planning(self.map)
                # self.planner.visualize_episode(cw)   #visualize 원하면 이거

                self.go_tracking = True
                
                # self.modecmd_msg.MorA = 0x00
                # self.modecmd_msg.Gear = 0x02
                # self.modecmd_msg.EStop = 0x00
                
                self.drivecmd_msg.KPH = 0
                self.drivecmd_msg.brake = 10
                
                # print("111111111111111111111111111111111111111111111111111111111")
                # 경로 생성 후 추종 클래스 선언
                # self.tracker = Tracking_DQN()
                
  
            # elif self.go_tracking and not self.tracker.tracking_End:
            #     # print("hihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihihhihihih")
            #     # 일단 최초 path 받은 시점에서 600,600으로 scaling 진행
            #     if not self.tracker.scaled :
            #         self.path_x, self.path_y = self.tracker.scaling(self.path_x, self.path_y)
                
            #     # print("x",self.path_x,"y :", self.path_y)
            #     self.modecmd_msg, self.drivecmd_msg  = self.tracker.stanley(self.path_x, self.path_y)
            #     rospy.loginfo("======DQN Tracking .. ")
                
            # if self.tracker.tracking_End :
            #     # tracking 끝난 후 변수 초기화 
            #     self.tracker.tracking_End = False
            #     self.map_make.trig = False
            #     self.tracker.scaled = False
            #     rospy.loginfo("=====================Tracking End !")

        # print(self.modecmd_msg)
        # print(self.drivecmd_msg)
        
        self.mode_pub.publish(self.modecmd_msg)
        self.drive_pub.publish(self.drivecmd_msg)
        
if __name__ == "__main__" :
    URP()
        