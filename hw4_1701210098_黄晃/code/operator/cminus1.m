function [c] = cminus1(a,b)
c = cellfun(@minus, a, b, 'UniformOutput', 0);
end