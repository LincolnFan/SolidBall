import numpy as np


def angle(x, y):  # 输入xy为列表即可
    x_a = np.array(x)
    y_a = np.array(y)
    l_x = np.sqrt(x_a.dot(x_a))
    l_y = np.sqrt(y_a.dot(y_a))
    dian_ = x_a.dot(y_a)
    cos_ = dian_/(l_x * l_y)
    angle_hu = np.arccos(cos_)
    angle_du = angle_hu*180/np.pi
    return angle_du
