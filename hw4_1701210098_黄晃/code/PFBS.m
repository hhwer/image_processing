function [u, snr] = PFBS(f1, f, A, W, WT, tol, kappa, L, lambda, maxstep)

u = f1;
alpha = W(u);
alpha_1 = alpha;
f1_norm = norm(f1, 'fro');
f_norm = norm(f, 'fro');
WAf1 = W(A(f1));
snr = f_norm / norm(f1 - f, 'fro');
for iter = 2:maxstep
    u1 = u;
    F2_grad = cplus2((cminus2(W(A(A(u))), WAf1)), ctimes2(cminus2(alpha, W(u)), kappa));
    g = cminus2(alpha, ctimes2(F2_grad, 1/L));
    alpha = Shrink(g, lambda/L, 2);
    u = WT(alpha);
    err = norm(u-u1, 'fro')/f1_norm;
    snr(iter) = f_norm / norm(u - f, 'fro');
    if err < tol
        snr = snr(1:iter);
        break;
    end
end
snr = 10*log10(snr.^2);
u = WT(alpha);
end

