function [gt_center_x, gt_center_y] = calculate_center_from_gt(label_file)
    content = fileread(label_file);
    parts = sscanf(content, '%f');
    if length(parts) < 5
        error('Invalid ground truth format in %s', label_file);
    end
    x = parts(2);
    y = parts(3);
    w = parts(4);
    h = parts(5);
    gt_center_x = x + w / 2;
    gt_center_y = y + h / 2;
end