function [cache] = cache_expire(cache)
    recent_threshold = 3;

    cache_keys = fieldnames(cache);
    [N, z] = size(cache_keys);
    
    for i = 1:N
        cache_key = cache_keys{i};
        used_recently = cache.(cache_key).count >= recent_threshold;

        % Delete infrequently used inversions
        if ~used_recently
            cache = rmfield(cache, cache_key);
        % For those not deleted, reset the "recent" count
        else
            cache.(cache_key).count = 0;
        end
    end

end