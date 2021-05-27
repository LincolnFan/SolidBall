from utils import angle, root_2, arc_tan_du
import cv2  # a library of programming functions mainly aimed at real-time computer vision
import mediapipe as mp  # open source cross-platform, customizable ML solutions for live and streaming media
import numpy as np
import sympy as sp
import configparser
import initParam


def SoildBallAdvice(url):
    config = configparser.ConfigParser()
    config.read('../docs/param_suggestion.ini')
    cap = cv2.VideoCapture(url)  # read our video
    with open('../docs/coco.names', 'r') as f:    # 模块前期配置
        classNames = f.read().splitlines()
    weightsPath = "../docs/frozen_inference_graph.pb"
    configPath = "../docs/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    pose = mp.solutions.pose.Pose()    # pose: the function to get the pose
    angleBackMax = 0
    angle_kneeMin = 180
    ball_position_x = []
    ball_position_y = []
    y_min = 10000
    while True:
        success, img = cap.read()   # this img is in bgr
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # bgr to rgb
        imgRGB = img    # no need to transfer
        try:
            results = pose.process(imgRGB)  # results.pose_landmarks mean the key point of our pose
            classIds, conf, bbox = net.detect(imgRGB, confThreshold=initParam.threshold)  # confidence
        except AttributeError:
            break
        bbox = list(bbox)  # list() 函数用于将元组、区间（range）、字典转换为列表
        conf = list(np.array(conf).reshape(1, -1)[0])
        conf = list(map(float, conf))
        indices = cv2.dnn.NMSBoxes(bbox, conf, initParam.threshold, initParam.nms_threshold)

        if results.pose_landmarks:
            lmList = []
            for pose_id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape  # height width channel
                cx, cy = lm.x * w, lm.y * h   # to get the true x,y of the landmarks
                lmList.append([cx, cy])

            # --求最大的后仰角度--
            if lmList:
                # y = [lmList[28][0] - lmList[27][0], lmList[28][1] - lmList[27][1]]
                y = [1, 0]
                if lmList[30][0] + lmList[29][0] > lmList[31][0] + lmList[32][0]:
                    direction = 1  # 'left'
                else:
                    direction = -1  # 'right'
                x_body = [lmList[11][0] + lmList[12][0] - lmList[23][0] - lmList[24][0],
                          lmList[11][1] + lmList[12][1] - lmList[23][1] - lmList[24][1]]     # 腰指向肩(躯干)的向量
                if direction * x_body[0] > 0:
                    back_angle = abs(90 - angle(x_body, y))
                    if 70 > back_angle > angleBackMax:  # 防止大的离谱？
                        angleBackMax = back_angle
                # 求出手点纵坐标
                y_m = lmList[17][1]/4 + lmList[18][1]/4 + lmList[18][1]/4 + lmList[18][1]/4
                if y_m < y_min:
                    y_min = y_m
                #   求膝盖弯曲角度
                if direction * (lmList[26][0]-lmList[25][0]) > 0:
                    thigh = [lmList[24][0] - lmList[26][0], lmList[24][1] - lmList[26][1]]
                    calf = [lmList[28][0] - lmList[26][0], lmList[28][1] - lmList[26][1]]
                else:
                    thigh = [lmList[23][0] - lmList[25][0], lmList[23][1] - lmList[25][1]]
                    calf = [lmList[27][0] - lmList[25][0], lmList[27][1] - lmList[25][1]]   # 选择后腿
                knee_angle = angle(thigh, calf)
                if knee_angle < angle_kneeMin:
                    angle_kneeMin = knee_angle
                # 求两脚分开的距离
                # dis_feet.append(abs(lmList[28][0] - lmList[27][0]))
        if len(classIds) != 0:
            for i in indices:
                i = i[0]
                if classNames[classIds[i][0] - 1] == 'sports ball':
                    box = bbox[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    if lmList:
                        dis1 = np.linalg.norm([lmList[15][0] + lmList[16][0] - 2 * x - w,  # distance ball to hand
                                               lmList[15][1] + lmList[16][1] - 2 * y - h], ord=2)  # 2范数
                        dis2 = np.linalg.norm([lmList[13][0]+lmList[14][0] - lmList[15][0] - lmList[16][0],  # 小臂
                                               lmList[13][1]+lmList[14][1] - lmList[15][1] - lmList[16][1]], ord=2)
                        if dis1 > 0.25 * dis2:
                            ball_position_x.append(x + w / 2)
                            ball_position_y.append(y+h/2)
    cap.release()
    cv2.destroyAllWindows()

    coe = np.polyfit(ball_position_x, ball_position_y, 2)   # use Parabola to fit the ball_track_point
    p_x = sp.symbols('p_x')
    p_y = coe[0] * p_x * p_x + coe[1] * p_x + coe[2]    # 拟合曲线
    x_r = root_2(coe[0], coe[1], coe[2] - y_min)
    ball_angle = arc_tan_du(sp.diff(p_y, p_x).evalf(subs={p_x: x_r[0]}))
    suggest = config.items('suggest1')
    string = ['', '身体可以后仰地更多一点', '身体后仰太多了', '球出手的角度可以更大一点', '球出手的角度可以稍微小一点', '发力时注意弯曲膝盖蹬地面']
    sug_dic = {}
    advice = ''
    for dic_i in suggest:
        sug_dic[dic_i[0]] = float(dic_i[1])
    if angleBackMax < sug_dic['back_angle_suggested'] - sug_dic['back_angle_tolerance']:
        advice = advice + string[1] + ','
    elif angleBackMax > sug_dic['back_angle_suggested'] + sug_dic['back_angle_tolerance']:
        advice = advice + string[2] + ','
    if ball_angle < sug_dic['ball_angle_suggested'] - sug_dic['ball_angle_tolerance']:
        advice = advice + string[3] + ','
    elif ball_angle > sug_dic['ball_angle_suggested'] + sug_dic['ball_angle_tolerance']:
        advice = advice + string[4] + ','
    if angle_kneeMin > sug_dic['knee_angle_suggested']:
        advice = advice + string[5]
    return advice
