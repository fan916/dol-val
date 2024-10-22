function difference = calculate_pixel_difference(pred_center, gt_center)
    difference = sqrt((pred_center(1) - gt_center(1))^2 + (pred_center(2) - gt_center(2))^2);
end