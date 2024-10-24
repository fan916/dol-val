import cv2
import numpy as np
import os
import time
import json
from sot_fast import process_image

def calculate_time_score(time_in_ms):
    """根据处理时间计算得分"""
    if time_in_ms <= 0.9:
        return 100
    elif time_in_ms >= 5:
        return 0
    return 60 + (100 - 60) * (5 - time_in_ms) / (5 - 0.9)

def calculate_acc_score(pixels):
    """根据像素差计算得分"""
    if pixels <= 1:
        return 100
    elif pixels >= 10:
        return 0
    return 60 + (100 - 60) * (10 - pixels) / (10 - 1)

def calculate_center_from_gt(label_file):
    """读取真值标签文件并计算框的中心坐标"""
    with open(label_file, 'r') as f:
        category, x, y, w, h = map(float, f.readline().strip().split())
        return x + w / 2, y + h / 2

def calculate_pixel_difference(pred_center, gt_center):
    """计算预测标签和真值标签中心坐标的像素差值"""
    return np.linalg.norm(np.array(pred_center) - np.array(gt_center))

def main():
    input_folder = r"E:\track-train\01" # 测试图片文件夹
    gt_folder = r"E:\track-train\02" # 真值标签的txt文件夹路径
    center_folder = r"E:\track-train\track-label" # 中心坐标结果保存文件夹
    result_log = r"E:\track-train\track-log.txt" # 保存计算的日志文件路径

    if not os.path.exists(center_folder):
        os.makedirs(center_folder)

    log_entries = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read image: {filename}")
                continue

            start_time = time.time()
            center = process_image(image)

            if center is not None:
                center_file_path = os.path.join(center_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(center_file_path, 'w') as f:
                    f.write(f"{center[0]:.2f} {center[1]:.2f}\n")

                elapsed_time = (time.time() - start_time) * 1000
                time_score = calculate_time_score(elapsed_time)

                gt_file = os.path.join(gt_folder, f"{os.path.splitext(filename)[0]}.txt")
                if os.path.exists(gt_file):
                    gt_center = calculate_center_from_gt(gt_file)
                    pixel_difference = calculate_pixel_difference(center, gt_center)
                    acc_score = calculate_acc_score(pixel_difference)

                    log_entry = {
                        "filename": filename,
                        "pixel_difference": pixel_difference,
                        "acc_score": acc_score,
                        "time_score": time_score,
                        "score": time_score * acc_score / 10000
                    }
                    log_entries.append(log_entry)
                    print(f"{filename}, time_score = {time_score}, acc_score = {acc_score}, score = {time_score * acc_score / 10000}")
                else:
                    print(f"Ground truth file for {filename} not found!")
            else:
                print(f"No center detected for image: {filename}")

    with open(result_log, "w") as log_file:
        json.dump(log_entries, log_file, indent=4)

if __name__ == "__main__":
    main()
