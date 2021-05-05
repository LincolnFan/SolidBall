import cv2  # a library of programming functions mainly aimed at real-time computer vision
import mediapipe as mp  # open source cross-platform, customizable ML solutions for live and streaming media
import time
import angle


class poseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True,     # initial
                 detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode  # default
        self.smooth = smooth
        self.upBody = upBody
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils     # to use that draw programme
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
        self.direction = 'right'

    def findPose(self, img, draw=True):
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # bgr to rgb
        imgRGB = img
        lmList = []
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = imgRGB.shape  # height width channel
                # print(id, lm)
                cx, cy, cz = int(lm.x*w), int(lm.y*h), int(lm.z*w)   # to get the true x,y of the landmarks
                lmList.append([id, cx, cy, cz])
            if draw:
                self.mpDraw.draw_landmarks(imgRGB, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                # draw landmark and connections on imgRGB
        return imgRGB, lmList

    def findBackAngel(self, img):   # 身体后仰角度
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                lmList.append([lm.x, lm.y, lm.z])
        if lmList:
            y = [lmList[28][0] - lmList[27][0], lmList[28][1] - lmList[27][1]]
            if lmList[30][0] + lmList[29][0] > lmList[31][0] + lmList[32][0]:
                self.direction = 'left'

            x = [lmList[11][0]+lmList[12][0]-lmList[23][0]-lmList[24][0],
                 lmList[11][1]+lmList[12][1]-lmList[23][1]-lmList[24][1]]
            if (self.direction == 'left' and x[0] > 0) or (self.direction == 'right' and x[0] < 0):
                return abs(90 - angle.angle(x, y))


def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')  # read our video
    pTime = 0  # previous time
    detector = poseDetector()   # 初始化
    while True:
        success, img = cap.read()  # this img is in bgr
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)    # 显示实时帧率
        cv2.imshow("Image", img)  # show the frames of the video
        cv2.waitKey(1)  # delay


if __name__ == "__main__":
    main()
