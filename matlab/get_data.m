function [endog, exog] = get_data(lag, delta)

    data = xlsread('recession probit data.xlsx',1);

    growth_rate = @(data, delta) (data ./ circshift(data, delta) - 1) * 100;

    sp500_return   = growth_rate(data(:, 4), delta);
    term_spread    = (data(:, 5) - data(:, 6));
    agg_emp_growth = growth_rate(data(:, 7), delta);
    agg_ip_growth  = growth_rate(data(:, 8), delta);
    state_columns  = growth_rate(data(:, 9:end), delta);

    growth_columns = circshift([ agg_emp_growth agg_ip_growth state_columns ], 1);

    endog = data(delta+lag+2:end, 2);
    exog = [ ones(size(sp500_return)) data(:, 3) sp500_return term_spread growth_columns ];
    exog = exog(delta+2:end-lag, :);

end