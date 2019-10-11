# -*-coding:utf-8 -*-

import cv2

INPUT_PATH = r"/media/wjl/0AA50DEE0AA50DEE/interview-scene-extraction-master/input/1.mp4"
OUTPUT_PATH = r"/media/wjl/0AA50DEE0AA50DEE/interview-scene-extraction-master/output"
START_POINT = 0
STOP_POINT = 30 # 分钟


videoCapture = cv2.VideoCapture(INPUT_PATH)
img_w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (img_w, img_h)  # 保存视频的大小
fps = videoCapture.get(cv2.CAP_PROP_FPS)  # cctv13 25fps

videoWriter = cv2.VideoWriter(OUTPUT_PATH + "/11.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

i, ret= 0, True
while ret:
    ret, frame = videoCapture.read()
    i += 1
    if (i >= 0 and i <= STOP_POINT*60*fps):
        videoWriter.write(frame)
    else:
        print("DONE")
        ret = False
videoWriter.release()
videoCapture.release()
