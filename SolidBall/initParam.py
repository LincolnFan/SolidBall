import cv2
threshold = 0.5  # Threshold to detect object 阈值
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
pose_draw = True  # whether to draw the pose
ball_draw = True  # whether to draw the ball
font = cv2.FONT_HERSHEY_PLAIN
color_blue = (255, 0, 0)    # color in bgr
color_red = (0, 0, 255)

