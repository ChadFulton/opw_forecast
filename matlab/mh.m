function [ys, gammas, rhos, accepts, cache] = mh( ...
    exog, endog, G0, G, sigma2, cache ...
)

    % Parameters
    [T, n] = size(exog);
    iterations = G0 + G + 1;
    I = 20;   % controls shape of y_rvs
    N = 100;  % controls number of periods y_rvs is drawn for
    cache_expire_periods = 500;

    % Cached arrays
    M0  = eye(T);
    M0s = M0 / sigma2;

    % Data arrays
    gammas  = zeros(n, iterations);
    rhos    = zeros(n, iterations);
    ys      = zeros(T, iterations);
    accepts = zeros(iterations, 1);
    gammas(1, :) = 1;

    % Random variates
    comparators = draw_rvs_comparators(iterations);
    gamma_rvs = draw_rvs_gamma(n, iterations);
    rho_rvs = draw_rvs_rho(n, iterations);

    % MH
    for t = 2:iterations
        % Conserve memory by drawing only y_rvs for N periods at a time
        l = mod(t-1, N);
        if l == 1
            y_rvs = draw_rvs_y(T, I, N);
        end
        
        % Cache operations
        if mod(t, cache_expire_periods) == 1
            cache = cache_expire(cache);
        end
        
        % Draw a Sample
        [ys(:, t), gammas(:, t), rhos(:, t), accepts(t), cache] = sample( ...
            exog, endog, M0, M0s, rhos(:, t-1), gammas(:, t-1),    ...
            y_rvs(:, :, l+1), gamma_rvs(t-1), rho_rvs(:, t-1),     ...
            comparators(t-1), sigma2, cache);
    end

end