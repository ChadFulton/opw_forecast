function [density, cache] = ln_mvn_density(M0, sigma2, y, exog, ...
                                           cache_key, cache)

    if isfield(cache, cache_key)
        Sigma = cache.(cache_key).Sigma;
        determinant = cache.(cache_key).determinant;
        cache.(key).count = cache.(cache_key).count + 1;
    else
        Sigma = inv(M0 + sigma2 * (exog * exog'));
        determinant = det(Sigma);
        
        cache.(cache_key) = struct;
        cache.(cache_key).Sigma = Sigma;
        cache.(cache_key).determinant = determinant;
        cache.(cache_key).count = 0;
    end
    
    density = -0.5 * log(determinant) - 0.5 * y' * Sigma * y;

end