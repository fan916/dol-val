# 跟踪算法&计算处理时间
import cv2
import numpy as np
import os
import time
import json

def calculate_score(time_in_ms):
    """
    根据处理时间计算得分:
    - 小于等于0.9ms: 100分
    - 等于5ms: 60分
    - 大于5ms: 0分
    """
    if time_in_ms <= 0.9:
        return 100
    elif time_in_ms >= 5:
        return 0
    else:
        # 根据 0.9ms 到 5ms 的线性插值计算得分
        # 100分降到60分，时间从0.9ms到5ms
        return 60 + (100 - 60) * (5 - time_in_ms) / (5 - 0.9)
    
    
def process_image(image):
    # 转换为灰度图像并模糊处理
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.blur(gray_frame, (3, 3))

    # 腐蚀操作
    kernel = np.ones((4, 4), np.uint8)
    processed_frame = cv2.erode(gray_frame, kernel)
    
    # 裁剪ROI区域
    roi_frame = processed_frame[processed_frame.shape[0] // 4: 3 * processed_frame.shape[0] // 4,
                                processed_frame.shape[1] // 4: 3 * processed_frame.shape[1] // 4]
    frame_little = roi_frame.copy()

    # 计算Scharr梯度
    g_scharrGradient_X = cv2.Scharr(frame_little, cv2.CV_16S, 1, 0)
    g_scharrGradient_Y = cv2.Scharr(frame_little, cv2.CV_16S, 0, 1)
    g_scharrAbsGradient_X = cv2.convertScaleAbs(g_scharrGradient_X)
    g_scharrAbsGradient_Y = cv2.convertScaleAbs(g_scharrGradient_Y)
    scharrImage = cv2.addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0)

    # 二值化处理
    _, binaryImg = cv2.threshold(scharrImage, 30, 255, cv2.THRESH_BINARY)

    # 计算质心位置
    totalWeightX = totalWeightY = totalWeight = 0
    step = 2

    for y in range(0, binaryImg.shape[0], step):
        for x in range(0, binaryImg.shape[1], step):
            weight = float(binaryImg[y, x])
            totalWeightX += x * weight
            totalWeightY += y * weight
            totalWeight += weight

    # 计算加权平均位置并绘制框
    centerX = centerY = None
    if totalWeight > 0:
        centerX = totalWeightX / totalWeight + processed_frame.shape[1] / 4
        centerY = totalWeightY / totalWeight + processed_frame.shape[0] / 4
        cv2.rectangle(image, (max(int(centerX - 30), 0), max(int(centerY - 30), 0)),
                      (min(int(centerX + 30), image.shape[1]), min(int(centerY + 30), image.shape[0])),
                      (0, 0, 255), 2)
        # 绘制框的中心点
        cv2.circle(image, (int(centerX), int(centerY)), 5, (0, 255, 0), -1)
    
    return (centerX, centerY)

def main():
    input_folder = r"E:\track-train\01"  # 输入文件夹路径
    center_folder = r"E:\track-label"  # 中心坐标输出文件夹路径
    log_file = r"E:\track-time-log.txt"  # 日志文件路径
    
    # 创建一个列表来存储所有记录
    log_entries = []

    if not os.path.exists(center_folder):
        os.makedirs(center_folder)
            
    for filename in os.listdir(input_folder):        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 根据需要支持的文件格式
            image_path = os.path.join(input_folder, filename)
            
            # 记录开始时间
            start_time = time.time()
            image = cv2.imread(image_path)

            if image is not None:
                center = process_image(image)

                # 保存绘制框的中心坐标到txt文件
                if center[0] is not None and center[1] is not None:
                    center_file_path = os.path.join(center_folder, f"{os.path.splitext(filename)[0]}.txt")
                    with open(center_file_path, 'w') as f:
                        f.write(f"{center[0]:.2f} {center[1]:.2f}\n")
                        
                # 记录结束时间
                end_time = time.time()
                
                # 计算并输出处理时间
                elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
                print(f"time:{elapsed_time}ms")
                #根据时间计算分数
                score = calculate_score(elapsed_time)
                print(f"score:{score}")
                
                # 将处理时间记录到日志文件
                # log.write(f"Processed {filename} in {elapsed_time:.2f} ms score:{score}\n")
                # 创建日志条目字典
                log_entry = {
                    "filename": filename,
                    "time": elapsed_time,
                    "score": score
                }
                log_entries.append(log_entry)

    # 在最后写入日志文件
    with open(log_file, "w") as log_file:
        json.dump(log_entries, log_file, indent=4)

if __name__ == "__main__":
    main()