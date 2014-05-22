function [y, gamma, rho, accept, cache] = sample(       ...
    exog, endog, M0, M0s, rho, gamma, y_rvs, gamma_rvs, ...
    rho_rvs, comparator, sigma2, cache                  ...
)
    % 1. Gibbs step: draw y
    gamma_indicators = logical(gamma);
    y = draw_y(                          ...
        rho(gamma_indicators), endog,    ...
        exog(:, gamma_indicators), y_rvs ...
    );

    % 2. Metropolis step: draw gamma and rho

    % Get the acceptance probability
    if gamma_rvs > 1
        gamma_star = draw_gamma(gamma, gamma_rvs);
        [prob_accept, cache] = calculate_accept(y, exog, M0, gamma, ...
                                                gamma_star, sigma2, cache);
    else
        gamma_star = gamma;
        prob_accept = 1;
    end

    % Update the arrays based on acceptance or not
    accept = prob_accept >= comparator;
    if accept
        rho = zeros(size(rho));
        gamma = gamma_star;
        % Draw rho
        gamma_indicators = logical(gamma);
        k_gamma = sum(gamma);
        rho(gamma_indicators) = draw_rho( ...
            M0s(1:k_gamma, 1:k_gamma),    ...
            y, exog(:, gamma_indicators), ...
            rho_rvs(1:k_gamma)            ...
        );
    end

end