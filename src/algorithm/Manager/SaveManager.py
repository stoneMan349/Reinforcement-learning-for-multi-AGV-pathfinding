"""
Draw pics and save pics
"""
import matplotlib.pyplot as plt
import torch
import os


def smoothing(p_data, length=10):
    smooth_data = []
    data_length = len(p_data)
    smooth_length = length
    for i in range(data_length-smooth_length+1):
        temp = p_data[i:i+smooth_length]
        smooth_data.append(sum(temp)/smooth_length)
    return smooth_data


def draw_picture(p_data, title="Title", x_label="x_label", y_label="y_label", color="g", save_path="", smooth=False):
    plt.figure(figsize=(16, 9))  # 调整长宽比
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if smooth:
        p_data = smoothing(p_data)
    plt.plot(p_data, color=color)
    plt.tight_layout()  # 去除白边
    if save_path != "":
        plt.savefig(save_path, dpi=300)  # 设置存储格式和分辨率
    # plt.show()


def save_logs(log_path, log_content):
    with open(log_path, 'w') as f:
        for one_log in log_content:
            f.write(str(one_log))
            f.write("\r\n")


