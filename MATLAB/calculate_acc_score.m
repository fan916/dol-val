function score = calculate_acc_score(pixels)
    if pixels <= 1
        score = 100;
    elseif pixels >= 10
        score = 0;
    else
        score = 60 + (100 - 60) * (10 - pixels) / (10 - 1);
    end
end