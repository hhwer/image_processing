    function [f1, f] =  add_noise_blur(f, noise_rate, kernel)
        if size(f,3) > 1
            f = rgb2gray(f);
        end
            f1 = imfilter(f, kernel,'replicate', 'same', 'conv');
            f1 = f1 + noise_rate*max(max(f))*randn(size(f1));
    end