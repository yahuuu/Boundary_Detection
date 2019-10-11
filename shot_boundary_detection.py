# -*-coding:utf-8 -*-

import cv2
import numpy as np
import time
import os
from faces_recognition import *

FILE_PATH = "/media/wjl/0AA50DEE0AA50DEE/interview-scene-extraction-master/input/11.mp4"
FRAME_GAP = 25  # 每秒取fps/FRAME_GAP帧
UP_THRESHOLD_OF_DIFF_SUM = 0.3
UP_THRESHOLD_OF_DIFF = 1.78


class shot_story_detection(object):
    def __init__(self,file_path=FILE_PATH):
        self.cap = cv2.VideoCapture(file_path)  # 返回一个capture对象
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # cctv13 25fps
        self.diff = {} # 可能作为筛选阈值的依据
        self.diff_sum = 0 # 存放求平均前的diff 累计和
        self.pre_hist = np.zeros((64, 64, 64))
        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.shot_boundary = {}
        self.shot_boundary_2frame = {}
        self.shot_boundary_face = {} #frame_index: (x,y,w,h)
        self.frame_index = 0  #每帧索引,也是开始的帧
        self.shot_diff = [] # 里面的ele和阈值比较

    def calt_hsv(self, frame):
        ################ 有一点我不明白，计算直方图之前需要进行图片归一化吗
        _hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        _hist = cv2.calcHist([_hsv], [0,1,2], None, [64,64,64], [0,256, 0,256, 0,256])
        _self_diff = np.sum(np.abs(self.pre_hist - _hist))
        # print("frame_index, cal_hsv", self.frame_index, _self_diff)
        if _self_diff/self.img_w/self.img_h > UP_THRESHOLD_OF_DIFF:
            self.shot_boundary_2frame[self.frame_index] = _self_diff

        self.diff_sum += _self_diff
        self.pre_hist = _hist
        # self.diff.append(_self_diff)
        self.diff[self.frame_index] = _self_diff
        # print(hist)
        # print(np.shape(hist))

    def chek_boundary(self):
        _n = self.frame_index // FRAME_GAP
        _hw = self.img_h * self.img_w
        _result = self.diff_sum / _hw / _n
        # print("frame_index, diff_sum, _result, _hw, _n", self.frame_index, self.diff_sum, _result, _hw, _n)
        self.shot_diff.append(_result)

        if _result > UP_THRESHOLD_OF_DIFF_SUM:
            #增加一个shot断点
            self.shot_boundary[self.frame_index] = _result
            #到达阈值后误差归零
            self.diff_sum = 0


    def record_face_detec_result(self, frame):
        """检测有没有人脸,并字典记录人脸帧和xywh"""
        #
        f = faces_recognizer(self.img_w, self.img_h)
        face_coord = f.online_recongnizion(self.frame_index, frame)
        # print(face_coord)
        if len(face_coord):
            self.shot_boundary_face[self.frame_index] = face_coord
            print(face_coord)

    def read_video(self):
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        #开始不检测边界
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)  # 设置要获取的帧号
        ret, frame= self.cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        self.calt_hsv(frame)
        while ret:
            #进入下个画面
            self.frame_index += FRAME_GAP
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)  # 设置要获取的帧号
            ret, frame= self.cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            if not ret:
                continue
            # 调用自己方法
            self.calt_hsv(frame)
            self.chek_boundary()
            ## 检测有没有人脸,记录
            self.record_face_detec_result(frame)

            # cv2.imshow('frame', frame) # 展示图片
            # cv2.waitKey(800)

            print("processing...%s"%self.frame_index)
        #检测完视频后,初始化
        # self.frame_index = 0
        self.cap.release()
        self.diff_sum = 0 # 存放求平均前的diff
        print("视频总时长%0.5f小时"%(self.frame_index/(25*60*60)))
        print(self.img_h," ", self.img_w)
        print(self.fps)
        # print("sorted_diff,",len(self.diff), sorted(self.diff.items(), reverse=True, key=lambda x:x[1]))
        print("帧切换大于阈值self_diff,", len(self.diff), sorted(self.diff.items(), reverse=True, key=lambda x:x[1]))
        print("一段累计的帧shot_diff  ,", len(self.shot_diff), sorted(self.shot_diff, reverse=True))
        print("依据一段累积帧找到的视频边界点shot_boundary,", len(self.shot_boundary), self.shot_boundary)
        print("依据两帧找到的边界img__boundary2, ", len(self.shot_boundary_2frame), self.shot_boundary_2frame)
        print("依据人脸检测找到的人脸frame_index:", self.shot_boundary_face)
        # print("unsorted_shot_diff", self.shot_diff)
        # self.pre_hist = np.zeros((64, 64, 64))



ssd = shot_story_detection()
ssd.read_video()