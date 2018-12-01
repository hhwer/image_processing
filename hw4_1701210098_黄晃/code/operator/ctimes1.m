function [c] = ctimes1(a,b)
%% a is a cell  b is a scalar

if length(size(a)) == 1
    for i = 1:size(a) 
        c{i} = a{i}*b;
    end
else
    for i = 1:size(a,1)
        for j =1:size(a,2)
            c{i,j} = a{i,j}*b;
        end
    end
end