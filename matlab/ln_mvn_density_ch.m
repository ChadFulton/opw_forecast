function [density, cache] = ln_mvn_density_ch(M0, sigma2, y, exog, ...
                                              cache_key, cache)
    
    if isfield(cache, cache_key)
        A = cache.(cache_key).A;
        determinant = cache.(cache_key).determinant;
        cache.(cache_key).count = cache.(cache_key).count + 1;
    else
        Sigma = M0 + sigma2 * (exog * exog');
    
        % Cholesky decomposition
        A = chol(Sigma, 'lower');

        % Determinant
        determinant = prod(diag(A))^2;
        
        cache.(cache_key) = struct;
        cache.(cache_key).A = A;
        cache.(cache_key).determinant = determinant;
        cache.(cache_key).count = 0;
    end
    
    % Solve linear system
    opts.LT = true;
    res = linsolve(A, y, opts);

    density = -0.5 * log(determinant) - 0.5 * (res' * res);

end