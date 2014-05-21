function [gamma_star] = draw_gamma(gamma, rvs)

    gamma_star = gamma;

    if rvs > 1
        if gamma_star(rvs) == 1
            gamma_star(rvs) = 0;
        else
            gamma_star(rvs) = 1;
        end
    end

end