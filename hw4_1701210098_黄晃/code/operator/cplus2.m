function [c] = cplus2(a,b)
c = cellfun(@cplus1, a, b, 'UniformOutput', 0);
end