function [u, snr] = TV( f1, f, A, tol, mu, delta, lambda, maxstep)
[u,snr] = ADMM(f1, f, A, @grad, @minus_div, tol, mu, delta, lambda, maxstep, 1);
end



function [g] = grad(x)
x_l = [x(:,1), x(:,1:end-1)];
x_r = [x(:,2:end), x(:,end)];
x_u = [x(1,:); x(1:end-1,:)];
x_d = [x(2:end,:); x(end,:)];
g{1} = (x_r - x_l)/2;
g{2} = (x_u - x_d)/2;
end

function [d] = minus_div(g)
x1_l = [g{1}(:,1), g{1}(:,1:end-1)];
x1_r = [g{1}(:,2:end), g{1}(:,end)];
x2_u = [g{2}(1,:); g{2}(1:end-1,:)];
x2_d = [g{2}(2:end,:); g{2}(end,:)];
d = (x1_r - x1_l)/2 + (x2_u - x2_d)/2;
d = -d;
end
