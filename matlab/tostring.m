function [str] = tostring(gamma)

    str = ['key' num2str(gamma)];
    str(isspace(str)) = '';

end