function [endog, exog] = get_data(lag, delta)

    data = xlsread('recession probit data.xlsx',1);

    sp500_return   = (data(:, 4) ./ circshift(data(:, 4), delta) - 1) * 100;
    term_spread    = (data(:, 5) - data(:, 6));
    agg_emp_growth = (data(:, 7) ./ circshift(data(:, 7), delta) - 1) * 100;
    agg_ip_growth  = (data(:, 8) ./ circshift(data(:, 8), delta) - 1) * 100;
    state_columns  = (data(:, 9:end) ./ circshift(data(:, 9:end), delta) - 1) * 100;

    growth_columns = circshift([ agg_emp_growth agg_ip_growth state_columns ], 1);

    endog = data(delta+lag+2:end, 2);
    exog = [ ones(size(sp500_return)) data(:, 3) sp500_return term_spread growth_columns ];
    exog = exog(delta+2:end-lag, :);

end