input_folder = 'E:\track-train\01'; % 测试图片文件夹
center_folder = 'E:\track-label'; % 中心坐标结果保存文件夹
log_file = 'E:\track-time-log.txt'; % 保存时间结果的日志文件路径
gt_folder = 'E:\track-train\02';  % 真值标签的txt文件夹路径
result_log = 'E:\track-acc-log.txt';  % 保存比较结果的日志文件路径


if ~exist(center_folder, 'dir')
    mkdir(center_folder);
end

log_entries = struct('filename', {}, 'time', {}, 'score', {});
total_time = 0;
total_images = 0;
total_accuracy = 0;

image_files = dir(fullfile(input_folder, '*.*'));

for k = 1:length(image_files)
    [~, ~, ext] = fileparts(image_files(k).name);
    if any(strcmpi(ext, {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}))
        image_path = fullfile(input_folder, image_files(k).name);
        image = imread(image_path);

        if ~isempty(image)
            start_time = tic; % 记录开始时间
            
            % 算法调用
            center = process_image(image); % 目标中心 (centerX, centerY)

            % 保存绘制框的中心坐标到txt文件
            if ~isempty(center)
                center_file_path = fullfile(center_folder, [image_files(k).name(1:end-4), '.txt']);
                fid = fopen(center_file_path, 'w');
                fprintf(fid, '%.2f %.2f\n', center(1), center(2));
                fclose(fid);
            end
            
            elapsed_time = toc(start_time) * 1000; % 转换为毫秒
            total_time = total_time + elapsed_time;
            total_images = total_images + 1;

            score = calculate_time_score(elapsed_time);
            total_accuracy = total_accuracy + score;

            log_entries(end+1) = struct('filename', image_files(k).name, 'time', elapsed_time, 'score', score);
        end
    end
end

% 计算平均处理时间
if total_images > 0
    average_time = total_time / total_images;
    fprintf('Average processing time: %.2f ms\n', average_time);

    % 计算平均时间分数
    average_accuracy = total_accuracy / numel(log_entries);
    fprintf('Average time score: %.2f\n', average_accuracy);

    % 在最后写入日志文件
    fid = fopen(log_file, 'w');
    json_entries = jsonencode(log_entries);
    fprintf(fid, '%s\n', json_entries);
    fclose(fid);
else
    fprintf('No valid images found for processing.\n');
end

log_entries_acc = struct('filename', {}, 'pixel_difference', {}, 'score', {});
total_accuracy = 0;

predicted_files = dir(fullfile(center_folder, '*.txt'));

for k = 1:length(predicted_files)
    predicted_file_name = predicted_files(k).name;
    predicted_file_name = strrep(predicted_file_name, '..', '.'); % 替换重复的点
    gt_file = fullfile(gt_folder, predicted_file_name);

    disp(gt_file);
    
    % 检查真值标签文件是否存在
    if exist(gt_file, 'file')
        % 读取真值标签文件
        gt_content = fileread(gt_file);
        if isempty(gt_content)
            fprintf('Ground truth file for %s is empty, skipping...\n', predicted_files(k).name);
            continue;
        end
        
        % 读取预测中心坐标
        pred_coords = sscanf(fileread(predicted_file), '%f')';
        if length(pred_coords) < 2
            fprintf('Predicted file %s is invalid, skipping...\n', predicted_file);
            continue;
        end
        pred_x = pred_coords(1);
        pred_y = pred_coords(2);
        
        % 读取真值标签中心坐标
        [gt_center_x, gt_center_y] = calculate_center_from_gt(gt_file);
        
        % 计算中心坐标的差值
        pixel_difference = calculate_pixel_difference([pred_x, pred_y], [gt_center_x, gt_center_y]);
        
        % 根据像素差计算得分
        score = calculate_acc_score(pixel_difference);
        total_accuracy = total_accuracy + score;
        
        % 创建日志条目字典
        log_entries_acc(end+1) = struct('filename', predicted_files(k).name, 'pixel_difference', pixel_difference, 'score', score);
    else
        fprintf('Ground truth file for %s not found!\n', predicted_files(k).name);
    end
end

% 计算平均精度分数
if numel(log_entries_acc) > 0
    average_accuracy = total_accuracy / numel(log_entries_acc);
    fprintf('Average accuracy score: %.2f\n', average_accuracy);

    % 在最后写入日志文件
    fid = fopen(result_log, 'w');
    json_entries_acc = jsonencode(log_entries_acc);
    fprintf(fid, '%s\n', json_entries_acc);
    fclose(fid);
else
    fprintf('No valid predictions found for accuracy evaluation.\n');
end
