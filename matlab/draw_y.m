function [y] = draw_y(rho, endog, exog, rvs)

    shape = size(exog);
    T = shape(1);
    shape = size(rvs);
    I = shape(2);
    max_iter = I*3;

    xB = exog * rho;
    y = rvs(:, 1) + xB;

    for t = 1:T
        i = 1;
        j = 1;
        if endog(t) == 1 && y(t) < 0
            rvs_ = rvs(t, :);
            while y(t) < 0
                % Increment
                i = i + 1;
                j = j + 1;
                % If we're not moving, just draw from the truncated normal
                if j > max_iter
                %     y(t) = stats.truncnorm.rvs(-xB(t), inf, loc=xB(t))
                    y(t) = truncnormrnd(1, xB(t), 1, 0, inf);
                    continue
                end
                % Make sure we have enough variates
                if i == I
                    rvs_ = randn(I,1);
                    i = 1;
                end
                % Set new value
                y(t) = xB(t) + rvs_(i);
            end
        elseif endog(t) == 0 && y(t) > 0
            rvs_ = rvs(t, :);
            while y(t) > 0
                % Increment
                i = i + 1;
                j = j + 1;
                % If we're not moving, just draw from the truncated normal
                if j > max_iter
                %     y(t) = stats.truncnorm.rvs(-inf, -xB(t), loc=xB(t))
                    y(t) = truncnormrnd(1, xB(t), 1, -inf, 0);
                    continue
                end
                % Make sure we have enough variates
                if i == I
                    rvs_ = randn(I,1);
                    i = 1;
                end
                % Set new value
                y(t) = xB(t) + rvs_(i);
            end
        end
    end
end