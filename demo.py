import cv2
import time
import PoseModule as pm
import os
import sys


sys.path.append(os.getcwd())
import ball_Tracker.Ball_Track as bt
cap = cv2.VideoCapture('PoseVideos/1.mp4')  # read our video
pTime = 0  # previous time
detector = pm.poseDetector()
tracker = bt.ballTracker()
angleMax = 0
while cap:
    try:
        success, img = cap.read()  # this img is in bgr
        try:
            position = tracker.Ball_Track(img, draw=True)
            if position:
                print(position)
            img, lmList = detector.findPose(img, draw=True)
            angle = detector.findBackAngel(img)
            if angle > angleMax:
                angleMax = angle
        except:
            pass
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS:'+str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.imshow("Image", img)  # show the frames of the video
        cv2.waitKey(1)  # delay
    except:
        break

print("后仰最大角={}".format(angleMax))
cap.release()
cv2.destroyAllWindows()
