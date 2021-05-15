from math import sqrt


def root_2(a, b, c):    # y_t means y_target
    delta = b*b - 4 * c * a
    if delta >= 0:
        x_r = [(-b-sqrt(delta))/(2 * a), (-b+sqrt(delta))/(2 * a)]
        return x_r
    else:
        return False
