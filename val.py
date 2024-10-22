import cv2
import numpy as np
import os
import time
import json


def calculate_time_score(time_in_ms):
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
 
def calculate_acc_score(pixels):
    """
    根据像素插值计算得分:
    - 小于等于1个像素: 100分
    - 等于10个像素: 60分
    - 大于10个像素: 0分
    """
    if pixels <= 1:
        return 100
    elif pixels >= 10:
        return 0
    else:
        return 60 + (100 - 60) * (10 - pixels) / (10 - 1) 
    
def calculate_center_from_gt(label_file):
    """
    读取真值标签文件并计算框的中心坐标
    标签格式：类别 左上角x 左上角y 宽 高
    """
    with open(label_file, 'r') as f:
        line = f.readline().strip().split()
        # 解析真值标签的各个部分
        category, x, y, w, h = map(float, line)
        # 计算真值框的中心坐标
        gt_center_x = x + w / 2
        gt_center_y = y + h / 2
    return gt_center_x, gt_center_y

def calculate_pixel_difference(pred_center, gt_center):
    """
    计算预测标签和真值标签中心坐标的像素差值
    """
    pred_x, pred_y = pred_center
    gt_x, gt_y = gt_center
    # 计算两个中心坐标的欧几里得距离（像素差值）
    difference = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
    return difference


def main():
    input_folder = r"E:\track-train\01" # 测试图片文件夹
    center_folder = r"E:\track-label" # 中心坐标结果保存文件夹
    log_file = r"E:\track-time-log.txt" # 保存时间结果的日志文件路径
    gt_folder = r"E:\track-train\02"  # 真值标签的txt文件夹路径
    result_log =  r"E:\track-acc-log.txt"  # 保存比较结果的日志文件路径
    
    predicted_folder = center_folder 
    log_entries = []
    
    total_time = 0
    total_images = 0
    total_accuracy = 0

    if not os.path.exists(center_folder):
        os.makedirs(center_folder)
            
    for filename in os.listdir(input_folder):        
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is not None:
                start_time = time.time()  # 记录开始时间               
                
                # 算法调用
                # 输入：单张图片 image
                # 返回：目标中心 (centerX, centerY)
                # eg.def process_image(image)
                #    .....
                #    return (centerX, centerY)
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
                total_time += elapsed_time
                total_images += 1
   
                score = calculate_time_score(elapsed_time)
                total_accuracy += score  # 累加精度得分

                # log.write(f"Processed {filename} in {elapsed_time:.2f} ms score:{score}\n")
                # 创建日志条目字典
                log_entry = {
                    "filename": filename,
                    "time": elapsed_time,
                    "score": score
                }
                log_entries.append(log_entry)

    # 计算平均处理时间
    average_time = total_time / total_images if total_images > 0 else 0
    print(f"Average processing time: {average_time:.2f} ms")
    # 计算平均时间分数
    average_accuracy = total_accuracy / len(log_entries) if log_entries else 0
    print(f"Average time score: {average_accuracy:.2f}")
    
    # 在最后写入日志文件
    with open(log_file, "w") as log_file:
        json.dump(log_entries, log_file, indent=4)
        
    # 创建一个列表来存储所有记录
    log_entries_acc = []
           
    for filename in os.listdir(predicted_folder):
        if filename.endswith('.txt'):
            predicted_file = os.path.join(predicted_folder, filename)
            gt_file = os.path.join(gt_folder, filename)

            # 检查真值标签文件是否存在
            if os.path.exists(gt_file):
                # 读取真值标签文件
                with open(gt_file, 'r') as f:
                    gt_content = f.readline().strip()
                
                # 检查真值标签内容是否为空
                if not gt_content:
                    print(f"Ground truth file for {filename} is empty, skipping...")
                    continue  # 跳过当前文件

                # 读取预测中心坐标
                with open(predicted_file, 'r') as f:
                    pred_x, pred_y = map(float, f.readline().strip().split())
                
                # 读取真值标签中心坐标
                gt_center_x, gt_center_y = calculate_center_from_gt(gt_file)
                
                # 计算中心坐标的差值
                pixel_difference = calculate_pixel_difference((pred_x, pred_y), (gt_center_x, gt_center_y))
                
                #根据像素插值计算得分
                score = calculate_acc_score(pixel_difference)
                total_accuracy += score  # 累加精度得分
                
                # 创建日志条目字典
                log_entry = {
                    "filename": filename,
                    "pixel_difference": pixel_difference,
                    "score": score
                }
                log_entries_acc.append(log_entry)
            else:
                print(f"Ground truth file for {filename} not found!")
                
    # 计算平均精度分数
    average_accuracy = total_accuracy / len(log_entries_acc) if log_entries_acc else 0
    print(f"Average accuracy score: {average_accuracy:.2f}")

    # 在最后写入日志文件
    with open(result_log, "w") as log_file:
        json.dump(log_entries_acc, log_file, indent=4)

if __name__ == "__main__":
    main()