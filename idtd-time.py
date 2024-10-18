import cv2
import numpy as np
import time
import math
import os
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

def new_ring_strel(ro, ri):
    d = 2 * ro + 1
    se = np.ones((d, d), dtype=np.uint8)
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0
    return se

def mnwth(img, delta_b, bb):
    img_f = img.copy()
    _, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    img_d = cv2.dilate(img, delta_b)
    img_e = cv2.erode(img_d, bb)

    out = cv2.subtract(img, img_e)
    out[out < 0] = 0
    return out

def move_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = mnwth(gray, delta_b, bb)
    return result

def my_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

if __name__ == "__main__":
    # 图片文件夹路径
    img_folder = r"E:\detect-train-1\01"
    center_folder = r"E:\detect-label"
    log_file = r"E:\detect-time-log.txt"  # 日志文件路径
    
    # 创建一个列表来存储所有记录
    log_entries = []
    
    os.makedirs(center_folder, exist_ok=True)

    ro = 11
    ri = 10
    delta_b = new_ring_strel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)
    
    # 获取图片文件列表
    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        
        #记录开始时间
        start_time = time.time()
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Failed to load image {img_file}")
            continue       

        result = move_detect(frame)

        # 二值化并展示
        _, binary_img = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)

        # 连接组件检测
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=4)

        # 找到最亮的点并画框
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc
        w, h = 20, 20
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
        
        # 保存中心点坐标到txt文件
        coord_file = os.path.join(center_folder, os.path.splitext(img_file)[0] + ".txt")
        with open(coord_file, 'w') as f:
            f.write(f"{x} {y}\n")
            
        # 记录结束时间
        end_time = time.time()
        
        # 计算并输出处理时间
        elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒

        #根据时间计算分数
        score = calculate_score(elapsed_time)
        
        # 创建日志条目字典
        log_entry = {
            "filename": img_file,
            "pixel_difference": elapsed_time,
            "score": score
        }
        log_entries.append(log_entry)
    # 在最后写入日志文件
    with open(log_file, "w") as log_file:
        json.dump(log_entries, log_file, indent=4)
