import cv2  # a library of programming functions mainly aimed at real-time computer vision
import mediapipe as mp  # open source cross-platform, customizable ML solutions for live and streaming media
import time

mpDraw = mp.solutions.drawing_utils     # use this to draw
mpPose = mp.solutions.pose
pose = mpPose.Pose()    # pose: the function to get the pose

cap = cv2.VideoCapture('../Videos/2.mp4')  # read our video
pTime = 0   # previous time

while True:
    try:
        success, img = cap.read()   # this img is in bgr
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # bgr to rgb
        imgRGB = img    # no need to transfer
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)   # results.pose_landmarks mean the key point of our pose
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape  # height width channel
                print(id, lm)
                cx, cy = int(lm.x_body * w), int(lm.y * h)   # to get the true x,y of the landmarks
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,  3, (255, 0, 0), 3)

        cv2.imshow("Image", img)    # show the frames of the video
        cv2.waitKey(1)  # delay
    except AttributeError:
        break
