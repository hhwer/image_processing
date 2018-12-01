function [c] = cminus2(a,b)
c = cellfun(@cminus1, a, b, 'UniformOutput', 0);
end