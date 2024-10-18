import os
import json

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
    predicted_folder = r"E:\track-label"  # 预测标签的txt文件夹路径
    gt_folder = r"E:\track-train\02"  # 真值标签的txt文件夹路径
    result_log =  r"E:\track-acc-log.txt"  # 保存比较结果的日志文件路径
    
    # 创建一个列表来存储所有记录
    log_entries = []
    
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
                score = calculate_score(pixel_difference)

                # 创建日志条目字典
                log_entry = {
                    "filename": filename,
                    "pixel_difference": pixel_difference,
                    "score": score
                }
                log_entries.append(log_entry)
            else:
                print(f"Ground truth file for {filename} not found!")

    # 在最后写入日志文件
    with open(result_log, "w") as log_file:
        json.dump(log_entries, log_file, indent=4)
                                
if __name__ == "__main__":
    main()
