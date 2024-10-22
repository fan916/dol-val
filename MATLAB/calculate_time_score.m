function score = calculate_time_score(time_in_ms)
    if time_in_ms <= 0.9
        score = 100;
    elseif time_in_ms >= 5
        score = 0;
    else
        score = 60 + (100 - 60) * (5 - time_in_ms) / (5 - 0.9);
    end
end