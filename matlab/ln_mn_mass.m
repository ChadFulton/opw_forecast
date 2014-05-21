function [prob] = ln_mn_mass(gamma)

    shape = size(gamma);
    prob = log(1/nchoosek(shape(1), sum(gamma)));

end