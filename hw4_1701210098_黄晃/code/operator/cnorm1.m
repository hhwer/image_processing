function [f] = cnorm1(c)
s = [c{:}];
f = norm(s, 'fro');
end