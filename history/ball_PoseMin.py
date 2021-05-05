import cv2  # a library of programming functions mainly aimed at real-time computer vision
import mediapipe as mp  # open source cross-platform, customizable ML solutions for live and streaming media
import numpy as np
import time
import angle

threshold = 0.5  # Threshold to detect object 阈值
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
classNames = []
with open('../dnn_DetectionModel/coco.names', 'r') as f:
    classNames = f.read().splitlines()
font = cv2.FONT_HERSHEY_PLAIN

color = (255, 0, 0)
color_red = (0, 0, 255)
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
angleMax = 0
cap = cv2.VideoCapture('../PoseVideos/2.mp4')  # read our video
pTime = 0   # previous time

while True:
    try:
        success, img = cap.read()   # this img is in bgr
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # bgr to rgb
        imgRGB = img    # no need to transfer
        results = pose.process(imgRGB)  # results.pose_landmarks mean the key point of our pose
        classIds, conf, bbox = net.detect(img, confThreshold=threshold)  # confidence
        bbox = list(bbox)  # list() 函数用于将元组、区间（range）、字典转换为列表
        conf = list(np.array(conf).reshape(1, -1)[0])
        conf = list(map(float, conf))
        indices = cv2.dnn.NMSBoxes(bbox, conf, threshold, nms_threshold)

        if results.pose_landmarks:
            draw = True  # whether to draw the pose
            if draw:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            lmList = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape  # height width channel
                cx, cy = int(lm.x*w), int(lm.y*h)   # to get the true x,y of the landmarks
                lmList.append([lm.x, lm.y])
                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

            if lmList:
                y = [lmList[28][0] - lmList[27][0], lmList[28][1] - lmList[27][1]]
                if lmList[30][0] + lmList[29][0] > lmList[31][0] + lmList[32][0]:
                    direction = 'left'
                else:
                    direction = 'right'
                x = [lmList[11][0] + lmList[12][0] - lmList[23][0] - lmList[24][0],
                     lmList[11][1] + lmList[12][1] - lmList[23][1] - lmList[24][1]]
                if (direction == 'left' and x[0] > 0) or (direction == 'right' and x[0] < 0):
                    back_angle = abs(90 - angle.angle(x, y))
                    if back_angle > angleMax:
                        angleMax = back_angle

        if len(classIds) != 0:
            ball_position = []
            draw = True  # whether to draw the ball
            for i in indices:
                i = i[0]
                if classNames[classIds[i][0] - 1] == 'sports ball':
                    box = bbox[i]
                    confidence = str(round(conf[i], 2))
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    if draw:
                        cv2.circle(img, (int(x+w/2), int(y+h/2)), int(w/4+h/4), color_red, thickness=2)
                    ball_position.append((int(x+w/2), int(y+h/2)))
                    print(ball_position)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), font,  3, (255, 0, 0), 3)

        cv2.imshow("Image", img)    # show the frames of the video
        cv2.waitKey(1)  # delay
    except:
        break
print("后仰最大角={}".format(angleMax))
cap.release()
cv2.destroyAllWindows()