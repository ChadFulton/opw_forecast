function [rvs] = draw_rvs_rho(n, iterations)

    rvs = mvnrnd(zeros(1,n)', eye(n), iterations)';

end