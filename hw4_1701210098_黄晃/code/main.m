function u = main(image_path, frame, Level, lambda, mu, delta,...
    kernel_size, gaussian_sigma, noise_rate, result_image_path)

if nargin<1, image_path = './image/fig1.png'; end 
if nargin<2, frame= 1; end % wavelet frame type
% 0--Haar; 1--piecewise linear; 3--piecewise cubic
if nargin<3, Level=2; end % decomposition level 1-4
if nargin<4, lambda = 0.05; end
if nargin<5, mu = 6; end
if nargin<6, delta = (5^(1/2)+1)/2; end
if nargin<7, kernel_size = 15; end
if nargin<8, gaussian_sigma = 1.5; end
if nargin<9, noise_rate = 0.1; end 
if nargin<10, result_image_path = './image/fig1_result.png'; end
addpath('./2D');
addpath('./operator')

kernel = fspecial('gaussian', [kernel_size, kernel_size], gaussian_sigma);
f = im2double(imread(image_path)); 
if size(f,3) > 1
    f = rgb2gray(f);
end
f1 = add_noise_blur(f, noise_rate, kernel);
[D,R]=GenerateFrameletFilter(frame);
W  = @(x) FraDecMultiLevel2D(x,D,Level);
WT = @(x) FraRecMultiLevel2D(x,R,Level);


A = @(u)(imfilter(u, kernel,'replicate', 'same', 'conv'));


[u1, snr1] = ADMM(f1, f, A, W, WT, 1e-5, mu, delta, lambda, 20, 2);
[u2, snr2] = PFBS(f1, f, A , W, WT, 1e-5, 8, 10, lambda, 20);
[u3, snr3] = TV(f1, f, A, 1e-5, mu, delta, lambda, 20);
% imshow(u);
path = './image/fig1_result.';
save_pic(u0, [result_image_path,'_admm0.png'],2);
save_pic(u1, [result_image_path,'_admm1.png'],2);
save_pic(u2, [result_image_path,'_admm2.png'],2);
save_err(err, './image/err_admm.png',3);
% plot(snr1,'LineWidth',4)
% hold on
% plot(snr2,'LineWidth',2)
% plot(snr3,'LineWidth',1)
% legend('admm','pfbs','tv')
end