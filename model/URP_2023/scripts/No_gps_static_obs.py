#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from morai_msgs.msg import CtrlCmd
from collections import deque

# Don't change
MAX_DISTANCE=1200
OBSTACLE_RADIUS=15
ACT_RAD=300
# Arc angle for detection
MIN_ARC_ANGLE=60
MAX_ARC_ANGLE=120
ANGLE_INCREMENT=4
# DETECTION_RANGE
DETECT_RANGE=180

# Default speed (km/h)
DEFAULT_VEL = 10


class Steer():
    def __init__(self):
        self.angle_lane = deque([0.0], maxlen = 50)
        self.trig_mission = False

        Width, Height = 600, 300
        self.warp_src = np.array([
            [300, 280],
            [300,150],
            [600, 280],
            [400,150]
        ], dtype=np.float32)

        #marker
        
        self.warp_dist = np.array([
            [300,300],
            [300, 0], #0
            [350,300],
            [350, 0]
        ], dtype=np.float32)
        
        self.warp_img_w = 600
        self.warp_img_h = 300
    
    def __call__(self, img, lidar, gps):
        if gps == 0:
            self.re_img = cv2.resize(img, (600,300))
            # cv2.imshow('re',self.re_img)
            warp_img, _, _ = self.warp_image(self.re_img, self.warp_src, self.warp_dist, (self.warp_img_w, self.warp_img_h))
            gray_img, zero_lane_trig = self.warp_process_image(warp_img, self.re_img)
            gray_img_roi = gray_img[:, 330:]
            gray_img_add = np.zeros((300, 330), dtype=np.uint8)
            gray_img = np.concatenate((gray_img_add, gray_img_roi), axis = 1)
            # print(gray_img.shape)
            # print(gray_img.shape) #(300, 600)
            # cv2.imshow('gray', gray_img)
            # Use front lidar data & make a local map
            lidar_raw_data = np.array(lidar[1:361]) * 40
            
            # Filter 'inf' value
            lidar_raw_data[lidar_raw_data>=MAX_DISTANCE] = -1
            current_frame = np.zeros((ACT_RAD, ACT_RAD * 2, 3), dtype=np.uint8)
            measured_points = []
            available_angles = []

            for i in range(len(lidar_raw_data)):
                # Skip empty points
                if lidar_raw_data[i] < 0: continue
                # Calculate xy points
                xy = [lidar_raw_data[i] * np.cos(np.deg2rad((i-180)/2)), lidar_raw_data[i] * np.sin(np.deg2rad((i-180)/2))]
                measured_points.append(xy)

            # Mark points on the map
            for point in measured_points:
                cv2.circle(current_frame, np.int32([ACT_RAD - np.round(point[1]), ACT_RAD - np.round(point[0])]), OBSTACLE_RADIUS, (255, 255, 255), -1)
            gray_img_cha = np.expand_dims(gray_img, axis = -1)
            gray_img_cur = np.concatenate((gray_img_cha,gray_img_cha,gray_img_cha), axis = 2)
            # print(gray_img_cur.shape)
            # print(current_frame.shape)
            current_frame = cv2.add(current_frame, gray_img_cur)
            # fk_img_cha = np.expand_dims(fk_canny_img, axis = -1)
            # fk_img_cur = np.concatenate((fk_img_cha,fk_img_cha,fk_img_cha), axis = 2)
            # fk_current_frame = cv2.add(current_frame, fk_img_c)
            # cv2.imshow("fk", fk_current_frame)
            # Draw a line to obstacles
            for theta in range(MIN_ARC_ANGLE-90, MAX_ARC_ANGLE-90, ANGLE_INCREMENT):
                # Find maximum length of line
                r = 1
                while r < DETECT_RANGE:
                    if current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][0] == 255 and\
                    current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][1] == 255 and\
                    current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][2] == 255: break
                    r += 1

                if r != DETECT_RANGE:
                    # draw a red line (detected)
                    cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1),(int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))), (0, 0, 255), 1)
                else:
                    # draw a gray line (not detected)
                    cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(theta))))), (0, 255, 0), 1)
                    available_angles.append(theta)

            # control angle
            if len(available_angles) == 0:
                middle_angle = 0
            else:
                middle_angle = np.median(np.array(available_angles), axis=0)
                # print(middle_angle)
            cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(middle_angle)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(middle_angle))))), (255, 255, 255), 1)

            # cv2.imshow("result", current_frame)
            # cv2.waitKey(1)
            
            if middle_angle < 3 and middle_angle > -3:
                middle_angle = 0
        else:
            middle_angle = 0
            self.trig_mission = True
        
        return middle_angle, self.trig_mission

    def color_filter(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        #original white
        lower = np.array([170, 150, 140])
        upper = np.array([255, 255, 255])

        #dark white
        # lower = np.array([50, 60, 80])
        # upper = np.array([255, 255, 255])
        
        # lower = np.array([40, 185, 10])
        # upper = np.array([255, 255, 255])

        self.lower_ylane = np.array([10, 100, 100])
        self.upper_ylane = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(hls, self.lower_ylane, self.upper_ylane)
        # cv2.imshow('yellow', yellow_mask)
        white_mask = cv2.inRange(image, lower, upper)
        # cv2.imshow('white', white_mask)
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        # masked = cv2.bitwise_and(image, image, mask = mask)
        
        return mask
        
    def warp_image(self, img, src, dst, size):
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
        # cv2.imshow("bird_img", warp_img)

        return warp_img, M, Minv
    
    def warp_process_image(self, img, canny_img):
        # fk_img = cv2.GaussianBlur(canny_img, (0, 0), 2)
        # fk_warp_img, _, _ = self.warp_image(fk_img, self.warp_src, self.warp_dist, (self.warp_img_w, self.warp_img_h))
        
        # fk_canny_img = cv2.Canny(fk_warp_img, 100, 120)
        # fk_lines = cv2.HoughLines(fk_canny_img, rho=1, theta = np.pi/180, threshold=50)
        # if fk_lines is not None:
        #     for line in fk_lines:
        #         rho, theta = line[0]
        #         a = np.cos(theta)
        #         b= np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * (a))
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * (a))
        #         self.angle_lane.append(a * 180/math.pi)

        #         cv2.line(fk_canny_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        # cv2.imshow('fk_img', fk_canny_img)
        

        #lane destroyed by 0
        zero_lane_trig = False
        canny_blurred_img = cv2.GaussianBlur(canny_img, (0, 0), 2)
        canny_w_f_img = self.color_filter(canny_blurred_img)
        canny_img = cv2.Canny(canny_w_f_img, 100, 120)
        canny_warp_img, _, _ = self.warp_image(canny_img, self.warp_src, self.warp_dist, (self.warp_img_w, self.warp_img_h))

        lines = cv2.HoughLines(canny_warp_img, rho=1, theta = np.pi/180, threshold=50)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b= np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                self.angle_lane.append(a * 180/math.pi)

                cv2.line(canny_warp_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        # print(b*180/math.pi)
        # cv2.imshow("bef_lane", canny_warp_img)
        
        #original lane
        blurred_img = cv2.GaussianBlur(img, (0, 0), 2)
        w_f_img = self.color_filter(blurred_img)
        w_f_img_cha = np.expand_dims(w_f_img, axis = -1)
        w_f_img_cur = np.concatenate((w_f_img_cha,w_f_img_cha,w_f_img_cha), axis = 2)
        # print(w_f_img.shape)
        grayscale = cv2.cvtColor(w_f_img_cur, cv2.COLOR_BGR2GRAY)
        ret, lane = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY) #170, 255
        # print(self.angle_lane)
        if lines is not None:
            for i in  self.angle_lane:
                if abs(i) < 30:
                    # print(i)
                    lane = np.zeros((ACT_RAD, ACT_RAD * 2), dtype=np.uint8)

        zero_lane = np.nonzero(lane)
        # print(len(zero_lane[0]))
        if len(zero_lane[0]) < 30:
            zero_lane_trig = True
        # cv2.imshow("aft_lane", lane)

        return lane, zero_lane_trig
