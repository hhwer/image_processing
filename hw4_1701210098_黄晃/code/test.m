function test(image_path, frame, Level, lambda, mu, delta,...
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
for frame = [0,1,3]
    [D,R]=GenerateFrameletFilter(frame);
    W  = @(x) FraDecMultiLevel2D(x,D,Level);
    WT = @(x) FraRecMultiLevel2D(x,R,Level);
    
    
    A = @(u)(imfilter(u, kernel,'replicate', 'same', 'conv'));
    
%     [u1, snr1] = ADMM(f1, f, A, W, WT, 1e-5, mu, delta, lambda, 20, 2);
%     [u2, snr2] = PFBS(f1, f, A , W, WT, 1e-5, 8, 10, lambda, 20);
%     [u3, snr3] = TV(f1, f, A, 1e-5, mu, delta, lambda, 20);
% % %     imshow(u);
% %     path = './image/fig1_result';
% %     save_pic(u1, [path,'_admm',num2str(frame),'.png'],2);
% %     save_err(snr1, ['./image/err_admm',num2str(frame),'.png'],3);
% %     save_pic(u2, [path,'_pfbs',num2str(frame),'.png'],2);
% %     save_err(snr2, ['./image/err_pfbs',num2str(frame),'.png'],3);
% %     save_pic(u3, [path,'_tv',num2str(frame),'.png'],2);
% %     save_err(snr3, ['./image/err_tv',num2str(frame),'.png'],3);
%     y = figure()
%     plot(snr1,'LineWidth',2)
%     hold on
%     plot(snr2,'LineWidth',2)
%     plot(snr3,'LineWidth',1)
%     xlabel('iterations','fontsize',15);
%     ylabel('SNR');
%     legend('admm','pfbs','tv')
%     path = ['./image/err_fram=',num2str(frame),'.png'];
%     print(y, path, '-dpng');
end
save_pic(f1,'fig1_noise.png',2);
end