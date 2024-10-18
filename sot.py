# 跟踪算法&计算处理时间
import cv2
import numpy as np
import os
import time
import argparse

def calculate_score(pixels):
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
        # 根据 0.9ms 到 5ms 的线性插值计算得分
        # 100分降到60分，时间从0.9ms到5ms
        return 60 + (100 - 60) * (10 - pixels) / (10 - 1)

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
    # 解析命令行参数
    parser = argparse.ArgumentParser('图像跟踪处理', add_help=False)
    parser.add_argument('--input_folder', default=r"E:\track-train\01", type=str, help="输入图像文件夹路径")
    parser.add_argument('--gt_folder', default=r"E:\track-train\02", type=str, help="真值标签文件夹路径")
    parser.add_argument('--center_folder', default=r"E:\track-label",type=str, help="中心坐标输出文件夹路径")    
    parser.add_argument('--log_file', default=r"E:\track-time-log.txt", type=str, help="处理时间日志文件路径")
    parser.add_argument('--result_log', default=r"E:\track-acc-log.txt", type=str, help="精度日志文件路径")
    
    args = parser.parse_args()
    
    input_folder = args.input_folder
    center_folder = args.center_folder
    gt_folder = args.gt_folder
    log_file = args.log_file
    result_log = args.result_log

    if not os.path.exists(center_folder):
        os.makedirs(center_folder)
            
    with open(log_file, 'w') as log:
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
                    
                    #根据时间计算分数
                    score = calculate_score(elapsed_time)
                    
                    # 将处理时间记录到日志文件
                    log.write(f"Processed {filename} in {elapsed_time:.2f} ms score:{score}\n")

    with open(result_log, 'w') as log:
        for filename in os.listdir(center_folder):
            if filename.endswith('.txt'):
                predicted_file = os.path.join(center_folder, filename)
                gt_file = os.path.join(gt_folder, filename)

                # 检查真值标签文件是否存在
                if os.path.exists(gt_file):
                    # 读取预测中心坐标
                    with open(predicted_file, 'r') as f:
                        pred_x, pred_y = map(float, f.readline().strip().split())
                    
                    # 读取真值标签中心坐标
                    gt_center_x, gt_center_y = calculate_center_from_gt(gt_file)
                    
                    # 计算中心坐标的差值
                    pixel_difference = calculate_pixel_difference((pred_x, pred_y), (gt_center_x, gt_center_y))
                    
                    #根据像素插值计算得分
                    score = calculate_score(pixel_difference)
                    
                    # 写入日志文件
                    log.write(f"{filename}: Pixel difference = {pixel_difference:.2f} pixels score:{score}\n")
                else:
                    print(f"Ground truth file for {filename} not found!")

if __name__ == "__main__":
    main()
