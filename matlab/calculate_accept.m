function [accept, cache] = calculate_accept(y, exog, M0, gamma, ...
                                            gamma_star, sigma2, cache)
    gamma_indicators = logical(gamma);
    gamma_star_indicators = logical(gamma_star);
    
    cache_key = tostring(gamma');
    [density, cache] = ln_mvn_density_ch(M0, sigma2, y, exog(:, gamma_indicators), cache_key, cache);
    denom = ln_mn_mass(gamma(2:end)) + density;
    
    cache_key = tostring(gamma_star');
    [density, cache] = ln_mvn_density_ch(M0, sigma2, y, exog(:, gamma_star_indicators), cache_key, cache);
    numer = ln_mn_mass(gamma_star(2:end)) + density;
    
    accept = exp(numer - denom);
    
end