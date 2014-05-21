function [rho] = draw_rho(M0, y, exog, rvs)

    M1 = inv(M0 + exog' * exog);
    m1 = M1 * exog' * y;
    A = chol(M1, 'lower');
    rho = m1 + A * rvs;

end