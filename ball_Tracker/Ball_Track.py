import numpy as np
import cv2


class ballTracker:
    def __init__(self, threshold=0.5, nms_threshold=0.2):
        self.threshold = threshold  # Threshold to detect object 阈值
        self.nms_threshold = nms_threshold  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
        self.dnnDM = cv2.dnn_DetectionModel

    def Ball_Track(self, img, draw=True):
        classNames = []

        with open('ball_Tracker/coco.names', 'r') as f:
            classNames = f.read().splitlines()  # import the classes of coco
        color = [0, 0, 255]
        weightsPath = "ball_Tracker/frozen_inference_graph.pb"     # the path set and net set, no need to change
        configPath = "ball_Tracker/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

        # with open('coco.names', 'r') as f:
        #     classNames = f.read().splitlines()  # import the classes of coco
        # color = [0, 0, 255]
        # weightsPath = "frozen_inference_graph.pb"     # the path set and net set, no need to change
        # configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

        net = self.dnnDM(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        # net.setPreferableTarget(36)
        classIds, conf, bbox = net.detect(img, confThreshold=self.threshold)  # confidence print(type(conf[0]))
        bbox = list(bbox)   # list() 函数用于将元组、区间（range）、字典转换为列表
        conf = list(np.array(conf).reshape(1, -1)[0])
        conf = list(map(float, conf))

        indices = cv2.dnn.NMSBoxes(bbox, conf, self.threshold, self.nms_threshold)
        if len(classIds) != 0:
            ball_position = []
            for i in indices:
                i = i[0]
                box = bbox[i]
                if classNames[classIds[i][0] - 1] == 'sports ball':
                    # confidence = str(round(conf[i], 2))
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    if draw:
                        cv2.circle(img, (int(x+w/2), int(y+h/2)), int(w/4+h/4), color, thickness=2)
                    ball_position.append((int(x+w/2), int(y+h/2)))
        return ball_position
def main():

    return


if __name__ == '__main__':
    main()
