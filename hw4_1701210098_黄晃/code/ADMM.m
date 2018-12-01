function [u, snr] = ADMM(f1, f, A, W, WT, tol, mu, delta, lambda, maxstep, c_deep)
if c_deep == 1
    cminus = @cminus1;
    cplus = @cplus1;
    ctimes = @ctimes1;
    cnorm = @cnorm1;
elseif c_deep == 2
    cminus = @cminus2;
    cplus = @cplus2;
    ctimes = @ctimes2;
    cnorm = @cnorm2;
end
    
u = f1;
d = W(u);
b = d;
ATf = A(f1);
shape = size(ATf);
A_cg = @(x)(reshape(A(A(reshape(x, shape)))+mu*WT(W(reshape(x, shape))), [], 1));
f1_norm = norm(f1, 'fro');
f_norm = norm(f, 'fro');
snr = f_norm / norm(f1 - f, 'fro');
for iter = 2:maxstep
    b_right = ATf + mu*WT(cminus(d,b));
    u = pcg(A_cg, reshape(b_right,[],1), 1e-6, 5);
    u(u > 1) = 1; 
    u(u < 0) = 0;
    u = reshape(u, shape);
    Wu = W(u);
    d = Shrink(cplus(Wu, b), lambda/mu, c_deep);
    b = cplus(b, ctimes(cminus(Wu, d), delta));
    err = cnorm(cminus(Wu, d))/f1_norm;
    snr(iter) = f_norm / norm(u - f, 'fro');
    if(err < tol)
        snr = snr(1:iter);
        break;
    end
end
snr = 10*log10(snr.^2);
end
