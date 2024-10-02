from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import threading
import time
import pyrealsense2 as rs
import numpy as np
import sys, Ui_test
import cv2 
import pyrealsense2 as rs
import random
import torch
import RobotControl_func_ros1
import math
import sys
from cv2 import aruco
import rospy

import gripper

aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

parameters = aruco.DetectorParameters()
#detector = aruco.ArucoDetector(aruco_dictionary, parameters)


model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
model.conf = 0.5
model.classes=[39,41]
rospy.init_node('my_node_name')
robot = RobotControl_func_ros1.RobotControl_Func()

g = gripper.Gripper(3)
g.gripper_reset()

INS = np.array([[606.777,0,321.633], #640x480
    [0,606.700,236.9529],
    [0,0,1]])
# INS = np.load("INS.npy") #1280x740
RC2G = np.load("Camera2Gripper.npy")[0:3,0:3]
TC2G = np.load("Camera2Gripper.npy")[:3,3]
POS_HOME = [375, -180, 450, -240, 0, 45]  # [375, -180, 350, -240, 0, 45]

[x,y,z,u,v,w] = POS_HOME

global state
state = 0
robot.set_TMPos(POS_HOME)


def fun_uv2XYZ(x,y,d):
    uv = np.array([x * d, y * d, 1 * d])
    xyz = np.linalg.inv(INS) @ uv
    XYZ = RC2G @ xyz + TC2G.flatten()
    return [-XYZ[1],XYZ[0],d]


def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        # cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        if dist > 0:
            distance_list.append(dist)

    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    if(len(distance_list) == 0):
        return mid_pos, 0
    else:
        return mid_pos, np.mean(distance_list)

def dectshow(org_img, boxs,depth_data):
    img = org_img.copy()
    if boxs.any():
        for box in boxs:
                # show rectangle
                # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            mid_pos, dist = get_mid_pos(org_img, box, depth_data, 24)
            if dist == 0:
                # print("Warning: Invalid distance returned from get_mid_pos.")
                continue

            
            radius = 5  # Example radius for the dot
            color = (0, 0, 255)  # Red color in BGR format
            thickness = -1  # Negative thickness indicates filled circle

            # Draw the dot on the image
            cv2.circle(img, (int(mid_pos[0]), int(mid_pos[1])), radius, color, thickness)

            # print distance on item
            # cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                        # (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow('dec_img', img)
        return mid_pos, dist, img

class myMainWindow(QMainWindow, Ui_test.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
       
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # Start streaming
        self.pipeline.start(self.config)

        img_thread = threading.Thread(target=self.get_image)
        # count_thread = threading.Thread(target=self.foo)
        img_thread.start()
        # count_thread.start()
        self.btn_show.clicked.connect(self.foo)
        self.pushButton_1.clicked.connect(self.BlackTea)
        self.pushButton_2.clicked.connect(self.Milk)
        self.pushButton_3.clicked.connect(self.MilkTea)
         
    def foo(self):
        self.pipeline.stop()
        sys.exit()

    #red(black tea) id 0
    #blue(coffee) id 1
    #yellow(milk) id 2
    def BlackTea(self):
        print("Pour Black Tea!")
        global state
        state = 1

    def Milk(self):
        print("Pour Milk!")
        global state
        state = 2

    def MilkTea(self):
        print ("Pour Milk Tea!")
        global state
        state = 3
        
    def get_image(self):
        global y2
        global state
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue
                
                gray_frame = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2GRAY)
                color_image = np.asanyarray(color_frame.get_data())
                
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
                aruframes = cv2.aruco.drawDetectedMarkers(image=color_image, corners=corners, ids=ids, borderColor=(0, 255, 0))

                depth_image = np.asanyarray(aligned_depth_frame.get_data())

                results = model(color_image)
                boxs = results.pandas().xyxy[0].values

                if state == 0:
                    try:
                        mid_pos, curr_dist, img = dectshow(color_image, boxs, depth_image)
                        # self.position_label.setText("[x, y, d]: [{:.0f}, {:.0f}, {:.2f} m]".format(mid_pos[0], mid_pos[1], curr_dist / 1000))
                        curr_pos = fun_uv2XYZ(mid_pos[0], mid_pos[1], curr_dist)
                        d = curr_pos[2] * math.sin(math.pi / 6)
                        x2 = curr_pos[2] * math.cos(math.pi / 6)
                        # self.trans_pos_label.setText("[x, y, z]: [{:.0f}, {:.0f}, {:.0f} ]".format(x2, curr_pos[1], d))
                    except Exception as e:
                        print(f"Error in state 0 processing: {e}")
                        img = color_image
                        pass
                    finally:
                        height, width, channel = img.shape
                        bytesPerline = channel * width 
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
                        canvas = QPixmap(height, width).fromImage(qimg)
                        self.label_rgb.setPixmap(canvas)
                        
                if state == 1:
                    state = 0
                    try:
                        if ids[0] == 0:
                            index = 0
                        else:
                            index = 1
                        print(index)
                        # print(boxs)
                        
                        meanX = (corners[index][0][0][0] + corners[index][0][1][0] + corners[index][0][2][0] + corners[index][0][3][0]) // 4
                        meanY = (corners[index][0][0][1] + corners[index][0][1][1] + corners[index][0][2][1] + corners[index][0][3][1]) // 4

                        if (boxs[index][0] <= meanX and boxs[index][2] >= meanX) and (boxs[index][1] <= meanY and boxs[index][3] >= meanY):
                            mid_pos, curr_dist = get_mid_pos(color_image, boxs[index], depth_image, 24)
                        else:
                            index = (index + 1) % 2
                            mid_pos, curr_dist = get_mid_pos(color_image, boxs[index], depth_image, 24)

                        print(index)

                        # show rectangle
                        cv2.rectangle(color_image, (int(boxs[index][0]), int(boxs[index][1])), (int(boxs[index][2]), int(boxs[index][3])), (0, 255, 0), 2)

                        img = color_image
                        height, width, channel = img.shape
                        bytesPerline = channel * width 
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
                        canvas = QPixmap(height, width).fromImage(qimg)
                        self.label_rgb.setPixmap(canvas)

                        self.position_label.setText("[x, y, d]: [{:.0f}, {:.0f}, {:.2f} m]".format(mid_pos[0], mid_pos[1], curr_dist / 1000))

                        curr_pos = fun_uv2XYZ(mid_pos[0], mid_pos[1], curr_dist)
                        d = curr_pos[2] * math.sin(math.pi / 6)
                        x2 = curr_pos[2] * math.cos(math.pi / 6)
                        
                        self.trans_pos_label.setText("[x, y, z]: [{:.0f}, {:.0f}, {:.0f} ]".format(x2, curr_pos[1], d))

                        y1 = y + curr_pos[1]+10
                        robot.set_TMPos([x+x2-120, y1, z-d+35, u, v, w])
                        time.sleep(1)
                        robot.set_TMPos([x+x2-55, y1, z-d+30, u+10, v, w])
                        time.sleep(1)
                        g.gripper_soft_off()
                        robot.set_TMPos([x+x2-60, y1, z-d+60, u, v, w])
                        robot.set_TMPos([890, -120, 390, -180, 0, 45])
                        time.sleep(1)
                        robot.set_TMPos([890, -120, 390, -140, 0, 45])
                        time.sleep(5)
                        robot.set_TMPos([750, y1, 500, -140, v, w])
                        robot.set_TMPos([x+x2-60, y1, z-d+150, u, v, w])
                        robot.set_TMPos([x+x2-60, y1, z-d+40, u, v, w])
                        g.gripper_on()
                        robot.set_TMPos([x+x2-30, y1, z-d+150, u, v, w])
                        robot.set_TMPos(POS_HOME)
                        print("Success!\n")
                    except Exception as e:
                        print(f"Error in state 1 processing: {e}")
                        img = color_image
                        pass

                if state == 2:
                    state = 0
                    try:
                        if ids[0] == 0:
                            index = 1
                        else:
                            index = 0
                        
                        # print(index)
                        # print(boxs)
                        # print(corners)
                        meanX = (corners[index][0][0][0] + corners[index][0][1][0] + corners[index][0][2][0] + corners[index][0][3][0]) // 4
                        meanY = (corners[index][0][0][1] + corners[index][0][1][1] + corners[index][0][2][1] + corners[index][0][3][1]) // 4

                        if (boxs[index][0] <= meanX and boxs[index][2] >= meanX) and (boxs[index][1] <= meanY and boxs[index][3] >= meanY):
                            mid_pos, curr_dist = get_mid_pos(color_image, boxs[index], depth_image, 24)
                        else:
                            index = (index + 1) % 2
                            mid_pos, curr_dist = get_mid_pos(color_image, boxs[index], depth_image, 24)

                        print(index)

                        # show rectangle
                        cv2.rectangle(color_image, (int(boxs[index][0]), int(boxs[index][1])), (int(boxs[index][2]), int(boxs[index][3])), (0, 255, 0), 2)
                        img = color_image
                        height, width, channel = img.shape
                        bytesPerline = channel * width 
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
                        canvas = QPixmap(height, width).fromImage(qimg)
                        self.label_rgb.setPixmap(canvas)

                        self.label_4.setText("[x, y, d]: [{:.0f}, {:.0f}, {:.2f} m]".format(mid_pos[0], mid_pos[1], curr_dist / 1000))

                        curr_pos = fun_uv2XYZ(mid_pos[0], mid_pos[1], curr_dist)
                        d = curr_pos[2] * math.sin(math.pi / 6)
                        x2 = curr_pos[2] * math.cos(math.pi / 6)

                        self.label_5.setText("[x, y, z]: [{:.0f}, {:.0f}, {:.0f} ]".format(x2, curr_pos[1], d))

                        y1 = y + curr_pos[1]+10

                        robot.set_TMPos([x+x2-120, y1, z-d+35, u, v, w])
                        time.sleep(1)
                        robot.set_TMPos([x+x2-55, y1, z-d+30, u+10, v, w])
                        time.sleep(1)
                        g.gripper_soft_off()
                        robot.set_TMPos([x+x2-60, y1, z-d+60, u, v, w])
                        robot.set_TMPos([890, -120, 390, -180, 0, 45])
                        time.sleep(1)
                        robot.set_TMPos([890, -120, 390, -140, 0, 45])
                        time.sleep(5)
                        robot.set_TMPos([750, y1, 500, -140, v, w])
                        robot.set_TMPos([x+x2-60, y1, z-d+150, u, v, w])
                        robot.set_TMPos([x+x2-60, y1, z-d+40, u, v, w])
                        g.gripper_on()
                        robot.set_TMPos([x+x2-30, y1, z-d+150, u, v, w])
                        robot.set_TMPos(POS_HOME)

                        # robot.set_TMPos([x+50, y1, z, u, v, w])
                        # time.sleep(0.5)
                        # robot.set_TMPos([x+x2-120, y1, z, u, v, w])
                        # time.sleep(0.5)
                        # robot.set_TMPos([x+x2-120, y1, z-d+35, u, v, w])
                        # time.sleep(1)
                        # robot.set_TMPos([x+x2-50, y1, z-d+35, u+10, v, w])
                        # time.sleep(1)
                        # g.gripper_soft_off()
                        # robot.set_TMPos([x+x2-60, y1, z-d+60, u, v, w])
                        # robot.set_TMPos([885, -120, 390, -180, 0, 45])
                        # time.sleep(1)
                        # robot.set_TMPos([885, -120, 390, -140, 0, 45])
                        # time.sleep(5)
                        # robot.set_TMPos([800, -120, 500, -140, v, w])
                        # robot.set_TMPos([x+x2-60, y1, z-d+150, u, v, w])
                        # robot.set_TMPos([x+x2-60, y1, z-d+40, u, v, w])
                        # g.gripper_on()
                        # robot.set_TMPos([x+x2-40, y1, z-d+125, u, v, w])
                        # robot.set_TMPos(POS_HOME)
                        print("Success!\n")
                    except Exception as e:
                        print(f"Error in state 2 processing: {e}")
                        img = color_image
                    pass

                if state == 3:
                    try:
                        if ids[0] == 0:
                            index = 1
                        else:
                            index = 0
                        
                        # print(index)
                        # print(boxs)
                        # print(corners)
                        meanX = (corners[index][0][0][0] + corners[index][0][1][0] + corners[index][0][2][0] + corners[index][0][3][0]) // 4
                        meanY = (corners[index][0][0][1] + corners[index][0][1][1] + corners[index][0][2][1] + corners[index][0][3][1]) // 4

                        if (boxs[index][0] <= meanX and boxs[index][2] >= meanX) and (boxs[index][1] <= meanY and boxs[index][3] >= meanY):
                            mid_pos, curr_dist = get_mid_pos(color_image, boxs[index], depth_image, 24)
                        else:
                            index = (index + 1) % 2
                            mid_pos, curr_dist = get_mid_pos(color_image, boxs[index], depth_image, 24)

                        print(index)

                        self.label_4.setText("[x, y, d]: [{:.0f}, {:.0f}, {:.2f} m]".format(mid_pos[0], mid_pos[1], curr_dist / 1000))

                        # show rectangle
                        cv2.rectangle(color_image, (int(boxs[index][0]), int(boxs[index][1])), (int(boxs[index][2]), int(boxs[index][3])), (0, 255, 0), 2)
                        img = color_image
                        height, width, channel = img.shape
                        bytesPerline = channel * width 
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
                        canvas = QPixmap(height, width).fromImage(qimg)
                        self.label_rgb.setPixmap(canvas)

                        curr_pos = fun_uv2XYZ(mid_pos[0], mid_pos[1], curr_dist)
                        d = curr_pos[2] * math.sin(math.pi / 6)
                        x2 = curr_pos[2] * math.cos(math.pi / 6)
                        
                        self.label_5.setText("[x, y, z]: [{:.0f}, {:.0f}, {:.0f} ]".format(x2, curr_pos[1], d))

                        y1 = y + curr_pos[1]+10

                        robot.set_TMPos([x+x2-120, y1, z-d+35, u, v, w])
                        time.sleep(1)
                        robot.set_TMPos([x+x2-55, y1, z-d+30, u+10, v, w])
                        time.sleep(1)
                        g.gripper_soft_off()
                        robot.set_TMPos([x+x2-60, y1, z-d+60, u, v, w])
                        robot.set_TMPos([890, -120, 390, -180, 0, 45])
                        time.sleep(1)
                        robot.set_TMPos([890, -120, 390, -140, 0, 45])
                        time.sleep(5)
                        robot.set_TMPos([750, y1, 500, -140, v, w])
                        robot.set_TMPos([x+x2-60, y1, z-d+150, u, v, w])
                        robot.set_TMPos([x+x2-60, y1, z-d+40, u, v, w])
                        g.gripper_on()
                        robot.set_TMPos([x+x2-30, y1, z-d+150, u, v, w])
                        robot.set_TMPos(POS_HOME)

                        # robot.set_TMPos([x+50, y1, z, u, v, w])
                        # time.sleep(0.5)
                        # robot.set_TMPos([x+x2-120, y1, z, u, v, w])
                        # time.sleep(0.5)
                        # robot.set_TMPos([x+x2-120, y1, z-d+35, u, v, w])
                        # time.sleep(1)
                        # robot.set_TMPos([x+x2-50, y1, z-d+35, u+10, v, w])
                        # time.sleep(1)
                        # g.gripper_soft_off()
                        # robot.set_TMPos([x+x2-60, y1, z-d+60, u, v, w])
                        # robot.set_TMPos([885, -120, 390, -180, 0, 45])
                        # time.sleep(1)
                        # robot.set_TMPos([885, -120, 390, -140, 0, 45])
                        # time.sleep(5)
                        # robot.set_TMPos([885, y1, 500, -140, v, w])
                        # robot.set_TMPos([x+x2-60, y1, z-d+150, u, v, w])
                        # robot.set_TMPos([x+x2-60, y1, z-d+40, u, v, w])
                        # g.gripper_on()
                        # robot.set_TMPos([x+x2-30, y1, z-d+150, u, v, w])
                        # robot.set_TMPos(POS_HOME)
                        print("Success!\n")
                        state = 1
                    except Exception as e:
                        print(f"Error in state 2 processing: {e}")
                        img = color_image
                    pass


        except Exception as e:
            print(f"Exit Program due to error: {e}")
            time.sleep(1)







if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())
