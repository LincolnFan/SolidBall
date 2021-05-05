import math

import cv2  # a library of programming functions mainly aimed at real-time computer vision
import mediapipe as mp  # open source cross-platform, customizable ML solutions for live and streaming media
import numpy as np
import sympy as sp
import time
import angle
from root2 import root_2

cap = cv2.VideoCapture('../PoseVideos/2.mp4')  # read our video
threshold = 0.5  # Threshold to detect object 阈值
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
pose_draw = True  # whether to draw the pose
ball_draw = True  # whether to draw the ball
font = cv2.FONT_HERSHEY_PLAIN
color_blue = (255, 0, 0)    # color in bgr
color_red = (0, 0, 255)
with open('../dnn_DetectionModel/coco.names', 'r') as f:    # 模块前期配置
    classNames = f.read().splitlines()
weightsPath = "../dnn_DetectionModel/frozen_inference_graph.pb"
configPath = "../dnn_DetectionModel/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

mpDraw = mp.solutions.drawing_utils     # use this to draw
mpPose = mp.solutions.pose
pose = mpPose.Pose()    # pose: the function to get the pose

pTime = 0   # previous time
angleMax = 0
ball_position_x = []
ball_position_y = []
y_min = 10000
while True:
    success, img = cap.read()   # this img is in bgr
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # bgr to rgb
    imgRGB = img    # no need to transfer
    try:
        results = pose.process(imgRGB)  # results.pose_landmarks mean the key point of our pose
        classIds, conf, bbox = net.detect(imgRGB, confThreshold=threshold)  # confidence
    except AttributeError:
        break
    bbox = list(bbox)  # list() 函数用于将元组、区间（range）、字典转换为列表
    conf = list(np.array(conf).reshape(1, -1)[0])
    conf = list(map(float, conf))
    indices = cv2.dnn.NMSBoxes(bbox, conf, threshold, nms_threshold)

    if results.pose_landmarks:
        if pose_draw:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        lmList = []
        for pose_id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape  # height width channel
            # cx, cy = int(lm.x*w), int(lm.y*h)   # to get the true x,y of the landmarks
            cx, cy = lm.x*w, lm.y*h
            lmList.append([cx, cy])

        if lmList:
            # y = [lmList[28][0] - lmList[27][0], lmList[28][1] - lmList[27][1]]
            y = [1, 0]
            if lmList[30][0] + lmList[29][0] > lmList[31][0] + lmList[32][0]:
                direction = 'left'
            else:
                direction = 'right'
            x = [lmList[11][0] + lmList[12][0] - lmList[23][0] - lmList[24][0],
                 lmList[11][1] + lmList[12][1] - lmList[23][1] - lmList[24][1]]     # 腰指向肩的向量
            if (direction == 'left' and x[0] > 0) or (direction == 'right' and x[0] < 0):
                back_angle = abs(90 - angle.angle(x, y))
                # print(back_angle)
                if 70 > back_angle > angleMax:  # 防止大的离谱？
                    angleMax = back_angle
            y_m = lmList[16][1]/2 + lmList[15][1]/2
            if y_m < y_min:
                y_min = y_m

    if len(classIds) != 0:
        for i in indices:
            i = i[0]
            if classNames[classIds[i][0] - 1] == 'sports ball':
                box = bbox[i]
                confidence = str(round(conf[i], 2))
                x, y, w, h = box[0], box[1], box[2], box[3]
                if ball_draw:
                    cv2.circle(img, (int(x+w/2), int(y+h/2)), int(w/4+h/4), color_red, thickness=2)
                if lmList:
                    dis1 = np.linalg.norm([lmList[15][0]+lmList[16][0] - 2 * x - w,
                                           lmList[15][1]+lmList[16][1] - 2 * y - h], ord=2)  # 2范数
                    dis2 = np.linalg.norm([lmList[13][0]+lmList[14][0] - lmList[15][0] - lmList[16][0],
                                           lmList[13][1]+lmList[14][1] - lmList[15][1] - lmList[16][1]], ord=2)
                    if dis1 > dis2:
                        ball_position_x.append(x+w/2)
                        ball_position_y.append(y+h/2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, 'Fps:'+str(int(fps)), (30, 30), font,  1.5, color_blue, 2)
    cv2.imshow("Image", img)    # show the frames of the video
    cv2.waitKey(1)  # delay

print("后仰最大角={}".format(angleMax))
cap.release()
cv2.destroyAllWindows()
print('球的轨迹横纵坐标：')
print(ball_position_x)
print(ball_position_y)
# print('球的出手点纵轴坐标：{}'.format(y_min))
coe = np.polyfit(ball_position_x, ball_position_y, 2)   # use Parabola to fit the ball_track_point
p_x = sp.symbols('p_x')
p_y = coe[0] * p_x * p_x + coe[1] * p_x + coe[2]
print('球的轨迹为：py = {}'.format(p_y))

x_r = root_2(coe[0], coe[1], coe[2] - y_min)
print('出球角度:{}'.format(float(math.atan(abs(sp.diff(p_y, p_x).evalf(subs={p_x: x_r[0]})))*180/math.pi)))   # 出球角度（maybe）

'''
y_min don't work well
may try to use the x when
 max the distance from foot to the hand 
'''