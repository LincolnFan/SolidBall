from math import sqrt
import numpy as np
import math


def root_2(a, b, c):    # y_t means y_target
    delta = b*b - 4 * c * a
    if delta >= 0:
        x_r = [(-b-sqrt(delta))/(2 * a), (-b+sqrt(delta))/(2 * a)]
        return x_r
    else:
        return False


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


def arc_tan_du(tan):
    return float(math.atan(abs(tan))*180/math.pi)
