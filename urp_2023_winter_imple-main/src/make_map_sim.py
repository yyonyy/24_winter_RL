#!/usr/bin/env python3

import torch 
import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *
from copy import deepcopy as dc
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, CompressedImage
from std_msgs.msg import String
from morai_msgs.msg import CtrlCmd
from sklearn.cluster import DBSCAN
from Models import ConvNet


# Reulst pixel size
PIXEL_WIDTH = 40
PIXEL_HEIGHT = 40

# left lane point
left_point = (8, 20)
left_point = (9,19)

# right lane point
right_point = (30, 20)
right_point = (29,19)

# Camera parameter
CAMERA_WIDTH = 600
CAMERA_HEIGHT = 300

# Lidar parameter
MAX_DISTANCE=1200
OBSTACLE_RADIUS=4
ACT_RAD=300
MIN_ARC_ANGLE=70
MAX_ARC_ANGLE=110
ANGLE_INCREMENT=4
# DETECT_RANGE=180
DETECT_RANGE=150 #200
CLUSTER_EPS = 15
CLUSTER_SAMPLE = 18

TRY_TIMES = 100
epochs = 50000
success_num = 0

class Show():
    def __init__(self):
        # rospy.init_node("hi")
        
        self.image = np.empty(shape=[0])
        self.combined_image = np.zeros((40,40,3),dtype=np.uint8)
        self.current_frame = np.zeros((ACT_RAD, ACT_RAD, 3), dtype=np.uint8)
        self.image_for_result = np.zeros((PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
        # self.resized_frame = np.zeros((PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
        
        self.bridge = CvBridge()
        
        self.lane_slope = None
        self.trig = False
        self.hi_param = 0
        self.saved_combined_image = None
        
        
        self.warp_src = np.array([
            [180, 300],
            [460,300],
            [0, 400],
            [640,400]
        ], dtype=np.float32)

        self.warp_dist = np.array([
            [CAMERA_WIDTH * (1/4) - 30,0],
            [CAMERA_WIDTH * (3/4) + 10, 0],
            [CAMERA_WIDTH * (1/4) - 10, CAMERA_HEIGHT],
            [CAMERA_WIDTH * (3/4),CAMERA_HEIGHT]
        ], dtype=np.float32)
        
        self.warp_img_w = 600
        self.warp_img_h = 300
        
        rospy.Subscriber("/image_jpeg_2/compressed", CompressedImage, self.img_callback)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
    
        # self.msg_pub = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
        
        # self.image_pub = rospy.Publisher("/combined_image", CompressedImage, queue_size=1)
        # self.trigger_pub = rospy.Publisher("/tracking", String, queue_size=10)
        
    def img_callback(self, data):
        image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        warp_img = self.warp_image(image, self.warp_src, self.warp_dist, (self.warp_img_w, self.warp_img_h))
        self.lane_img, self.lane_slope = self.warp_process_image(warp_img)
        
    def lidar_callback(self, data):
        self.lidar_raw_data = np.array(data.ranges[1:361]) * 30
        resized_frame = self.lidar_processing()
        # self.resized_frame = self.lidar_processing()
        if resized_frame is None:
            resized_frame = np.zeros((PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
        # else:
            # print("1111111111111111111111111111111111111111111111111")
        # print("?aaaaaaaaaaaaaaaaaaaaaaaaaaaaa", self.resized_frame)
        # self.main(self.resized_frame)
        self.main(resized_frame)
    
    def color_filter(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        # white
        lower = np.array([160, 120, 140])
        upper = np.array([255, 255, 255])

        # yellow
        self.lower_ylane = np.array([10, 100, 100])
        self.upper_ylane = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(hls, self.lower_ylane, self.upper_ylane)
        # cv2.imshow('yellow', yellow_mask)
        white_mask = cv2.inRange(image, lower, upper)
        # cv2.imshow('white', white_mask)
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        # mask = cv2.bitwise_and(image, image, mask = white_mask)
        # cv2.imshow('mask', mask)
        
        return mask
    
    def warp_image(self, img, src, dst, size):
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
        # cv2.imshow("ori", img)
        # cv2.imshow("warp", warp_img)
        # cv2.waitKey(1)
        
        return warp_img
    
    def warp_process_image(self, img):
        slope_list = []
        
        blurred_img = cv2.GaussianBlur(img, (0, 0), 2)
        w_f_img = self.color_filter(blurred_img)
        # grayscale = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
        # ret, lane = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY) #170, 255
        ret, lane = cv2.threshold(w_f_img, 200, 255, cv2.THRESH_BINARY) #170, 255
        # cv2.imshow("hi",lane)
        # cv2.waitKey(1)
        canny_img = cv2.Canny(lane, 60, 80)
        # cv2.imshow("hi",canny_img)
        # cv2.waitKey(1)
        lines = cv2.HoughLines(canny_img, 1, np.pi/180, 80, None, 0, 0)
        hough_img = canny_img.copy()
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int (x0 + 1000*(-b))
                y1 = int ((y0) + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                slope = 90 - degrees(atan(float(b / a)))
                
                slope_list.append(slope)
                
                if abs(slope) < 40:
                    cv2.line(hough_img, (x1, y1), (x2, y2), 0, 30)
                
                else:    
                    cv2.line(hough_img, (x1, y1), (x2, y2), 255, 8)
                    
        # cv2.imshow('Hough', hough_img)
        # cv2.waitKey(1)
        
        slope_to_plot = np.mean(slope_list)
        # print(slope_to_plot)
        
        return hough_img, slope_to_plot
    
    def lidar_processing(self):
        self.lidar_raw_data[self.lidar_raw_data>=MAX_DISTANCE] = -1
        current_frame = self.current_frame
        # current_frame_only_clustering = self.resized_frame
        current_frame_only_clustering = np.zeros((PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
        measured_points = []
        available_angles = []
        converted_points = []
        
        
        for i in range(len(self.lidar_raw_data)):
            # Skip empty points
            if self.lidar_raw_data[i] < 0: 
                continue
                # Calculate xy points
                # xy = [self.lidar_raw_data[i] * np.cos(np.deg2rad((i-180)/2)), self.lidar_raw_data[i] * np.sin(np.deg2rad((i-180)/2))]
            xy = [self.lidar_raw_data[i] * np.cos(np.deg2rad((i-180))), self.lidar_raw_data[i] * np.sin(np.deg2rad((i-180)))]
            # print(xy)
            measured_points.append(xy)
            
            # Mark points on the map
            for point in measured_points:
                transformed_point = [ACT_RAD/2 - np.round(point[1]), ACT_RAD - np.round(point[0])]
                converted_points.append(transformed_point)
                cv2.circle(current_frame, np.int32(transformed_point), OBSTACLE_RADIUS, (255, 255, 255), -1)

        if not converted_points:
            return
        
        # converted_points = np.nan_to_num(converted_points)
        
        dbscan = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_SAMPLE)
        labels = dbscan.fit_predict(converted_points)

        cluster_centers = []
        for label in set(labels):
            if label != -1:  # Ignore noise points
                cluster_points = np.array(converted_points)[np.array(labels) == label]
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)

        for center in cluster_centers:
            new_center_x = (center[0] / (ACT_RAD)) * PIXEL_WIDTH
            new_center_y = (center[1] / ACT_RAD) * PIXEL_HEIGHT

            cv2.circle(current_frame, (int(center[0]), int(center[1])), OBSTACLE_RADIUS, (0, 0, 255), -1)
            # cv2.circle(current_frame, (int(center[0]), int(center[1])), OBSTACLE_RADIUS, (255, 0, 255), -1)
            # cv2.circle(current_frame_only_clustering, (int(new_center_x), int(new_center_y)), 1, (0, 0, 255), -1)
            
            # 패딩 추가 부분
            padding = 1 #1
            cv2.rectangle(
                current_frame_only_clustering,
                (int(new_center_x) - 1 - padding, int(new_center_y) - 1 - padding),
                (int(new_center_x) + 1 + padding, int(new_center_y) + 1 + padding),
                (255, 0, 0),
                1
            )
            cv2.rectangle(current_frame_only_clustering, (int(new_center_x)-1, int(new_center_y)-1),(int(new_center_x)+1, int(new_center_y)+1), (1, 0, 0), cv2.FILLED)
    

        # Draw a line to obstacles
        for theta in range(MIN_ARC_ANGLE-90, MAX_ARC_ANGLE-90, ANGLE_INCREMENT):
            # Find maximum length of line
            r = 1
            while r < DETECT_RANGE:
                if current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD/2-1 - np.round(r * np.sin(np.deg2rad(theta))))][0] == 255 and\
                   current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD/2-1 - np.round(r * np.sin(np.deg2rad(theta))))][1] == 255 and\
                   current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD/2-1 - np.round(r * np.sin(np.deg2rad(theta))))][2] == 255: break
                r += 1

            if r != DETECT_RANGE:
                # draw a red line (detected)
                cv2.line(current_frame, (int(ACT_RAD/2)-1, ACT_RAD-1),(int(int(ACT_RAD/2)-1 - np.round(r * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))), (0, 0, 255), 1)
                self.trig = True
                # cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1),(int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))), (0, 0, 255), 1)
            else:
                # draw a gray line (not detected)
                cv2.line(current_frame, (int(ACT_RAD/2)-1, ACT_RAD-1), (int(int(ACT_RAD/2)-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(theta))))), (0, 255, 0), 1)
                # cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(theta))))), (0, 255, 0), 1)
                available_angles.append(theta)
        
        # # control angle
        # if len(available_angles) == 0:
        #     middle_angle = 0
        # else:
        #     middle_angle = np.median(np.array(available_angles), axis=0)
        #     print(middle_angle)        
        # cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(middle_angle)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(middle_angle))))), (255, 255, 255), 1)

        # non_black_condition = np.any(current_frame_only_clustering != [0, 0, 0], axis=-1)

        # Update non-black pixels to (1, 0, 0)
        # current_frame_only_clustering[non_black_condition] = [1, 0, 0]  
        

        cv2.imshow("lidar_result", current_frame)
        # cv2.imshow("lidar_result_clustering", current_frame_only_clustering)
        
        # resized_frame = cv2.resize(current_frame, (40, 40), interpolation=cv2.INTER_LINEAR)
        # resized_frame_only_clustering = cv2.resize(current_frame_only_clustering, (40, 20), interpolation=cv2.INTER_LINEAR)

        # cv2.imshow("lidar_result_resized", resized_frame)
        # cv2.imshow("clustering_result_resized", current_frame_only_clustering)
        cv2.waitKey(1)
        
        
        # if self.trig:
        #     # Save a copy of combined_image when trig is True
        #     self.saved_combined_image = dc(self.combined_image)
        #     self.hi_param += 1
        #     self.trig = False  # Reset trig to False after saving the image
        
        
        # return resized_frame
        return current_frame_only_clustering
        
    def main(self, resized_frame):
        # print('===map main !!')
        result_image = np.zeros((PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
        
        half_height = PIXEL_HEIGHT // 2

        # print(self.lane_slope)
        if self.lane_slope is None:
            self.lane_slope = 90
            
        slope = np.tan(np.radians(self.lane_slope))
        # print(np.radians(self.lane_slope))
        # print(slope)
        """
        차선 반대로 그려지길래 수정했습니다.
        """
        # left_end_x = int(left_point[0] + half_height / slope)
        # left_end_y = int(left_point[1] + half_height)
        
        # right_end_x = int(right_point[0] + half_height / slope)
        # right_end_y = int(right_point[1] + half_height)
        
        # start_point1 = (left_point[0], left_point[1] - half_height)
        # end_point1 = (left_end_x, left_end_y-1)
        
        # start_point2 = (right_point[0], right_point[1] - half_height)
        # end_point2 = (right_end_x, right_end_y-1)
        
        # cv2.line(result_image, start_point1, end_point1, (255, 255, 255), 1)
        # cv2.line(result_image, start_point2, end_point2, (255, 255, 255), 1)
        # start = (39,int((end_point1[0]+end_point2[0])/2))
        
        # 차선 그리기 수정 !
        left_start_x = int(left_point[0] - left_point[0] / slope)
        left_start_y = 40 - 1

        left_end_x = int(left_point[0] + left_point[0] / slope)
        left_end_y = 0
        
        right_start_x = left_start_x + int(40 / 2)
        right_start_y = 40 - 1
        
        right_end_x = left_end_x + int(40 / 2)
        right_end_y = 0        
        
        cv2.line(result_image, (left_end_x,left_end_y),(left_start_x,left_start_y),
                 (2, 0, 0), 1)
        
        cv2.line(result_image, (right_end_x,right_end_y),(right_start_x, right_start_y), 
                 (2, 0, 0), 1)
        
        start = (39,int((left_start_x+right_start_x)/2))
        
        # cv2.line(result_image, start_point1, end_point1, (1, 0, 0), 1)
        # cv2.line(result_image, start_point2, end_point2, (1, 0, 0), 1)

        # print("rr",result_image.shape)
        # print(resized_frame.shape)
        self.combined_image = cv2.add(result_image, resized_frame)
        
        # finish = (0,int((start_point1[0]+start_point2[0])/2))
        finish_range = range(int((left_end_x + right_end_x) / 2) - 2, int((left_end_x + right_end_x) / 2) + 3)
        finish_points = [(0, x) for x in finish_range]
        self.combined_image[start[0],start[1],1] = 10.0
        # self.combined_image[finish[0],finish[1],2] = 10.0
        
        for point in finish_points:
            if point[1] == int((left_end_x + right_end_x) / 2):
                self.combined_image[point[0], point[1], 2] = 20.0
                self.combined_image[point[0]+1, point[1], 2] = 20.0
                self.combined_image[point[0]+2, point[1], 2] = 20.0
                # print("point", point[0], point[1], grid[point[0], point[1], 2])
            else:
                self.combined_image[point[0], point[1], 2] = 10.0
                self.combined_image[point[0]+1, point[1], 2] = 10.0
                self.combined_image[point[0]+2, point[1], 2] = 10.0
                # print("1", point[0], point[1], grid[point[0], point[1], 2])
                
        # if self.trig:
        #     self.hi_param += 1
        self.current_frame = np.zeros((ACT_RAD, ACT_RAD, 3), dtype=np.uint8)

        # cv2.imshow("a",self.combined_image)
        # cv2.waitKey(1)
        
        # if self.trig:
        #     self.trig = False
        # condition = np.all(combined_image == [1, 0, 0], axis=-1)
        # combined_image[condition] = [0, 0, 255]
        # if self.trig and self.hi_param == 1:
        #     a = dc(self.combined_image)

        #     cv2.imshow("Combined Image When Trigger", result_image)
        #     cv2.imshow("Combined Image When Tger", resized_frame)
        #     cv2.waitKey(1)

        # return combined_image

        