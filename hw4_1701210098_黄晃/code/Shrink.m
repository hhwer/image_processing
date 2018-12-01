function [y] = Shrink(c, tau, c_deep)
shrink = @(v)sign(v).*max(abs(v)-tau, 0);
if c_deep == 1
    y = cellfun(shrink, c,  'UniformOutput', false);
elseif c_deep == 2
    y = cellfun(@(x)cellfun((shrink), x, 'UniformOutput', false), ...
        c, 'UniformOutput', false);
end

end

