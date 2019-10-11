# -*-coding:utf-8 -*-

import cv2
import numpy

SAVE_PATH = "/media/wjl/0AA50DEE0AA50DEE/testpic/"

class faces_recognizer(object):
    def __init__(self, video_w, video_h):
        self.img = None
        self.video_w = video_w
        self.video_h = video_h

    def read_local_img(self, img_path):
        self.img = cv2.imread(img_path)

    def save_img(self, filename, img):
        cv2.imwrite(filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])

    def reproc_face_coord(self, face_coord):
        """播音员头像判断，w,h不能太大，头像不能超过两个,位置不能太偏"""
        #超过两个头像， 或者没有头像
        if len(face_coord) > 3 or len(face_coord) < 1:
            return False

        for x, y, w, h in face_coord:
            #位置太偏
            if (y / self.video_h) > 0.26 or (y/ self.video_h) < 0.12:
                return False
            #脸太小
            if (w / self.video_w) < 0.12:
                return False
            #脸太大,w` about 0.156, h` about 0.2
            if (w / self.video_w) > 0.24 or (h / self.video_h) > 0.29:
                return False
        return True

    def online_recongnizion(self, frame_index, frame_img=()):
        self.img = frame_img if len(frame_img) else self.img  # 本地图片或者视频流
        # self.gray_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier("/home/wjl/work_task/new_segmentation/Classifier/haarcascade_frontalface_alt.xml")
        face_coord = face_detector.detectMultiScale(self.img)

        if self.reproc_face_coord(face_coord):
            for x, y, w, h in face_coord:
                cv2.rectangle(self.img, (x, y), (x+w, y+h), (0,0,255), 3)
                # cv2.imshow("result", self.img) # 4test
            self.save_img(SAVE_PATH + "%d.png"%frame_index, self.img)

            # cv2.waitKey(1000)
            cv2.destroyAllWindows()

            return face_coord # x, y, w, h
        else:
            cv2.destroyAllWindows()
            return ()




if __name__ == "__main__":
    img_path = "/home/wjl/Pictures/anchor/e.png"
    f = faces_recognizer()
    f.read_local_img(img_path)
    print(f.online_recongnizion(123, f.img))