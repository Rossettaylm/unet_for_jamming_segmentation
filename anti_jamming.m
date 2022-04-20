%%% anti-jamming
%%% date: 2022年3月2日


%% 得到干扰位置的索引，为一个python list
% pyrun("import os")
% pyrun("from PIL import Image")
% pyrun("from unet import Unet")
% pyrun("unet = Unet()")
% pyrun("image_path = './img/test658.png'")
% jamming_pos = pyrun("jamming_pos = unet.get_jamming_pos(unet, image_path)", "jamming_pos");
% 
% %% 将python list转为矩阵
% pic_size = [658, 877];
% matrix_j = zeros(pic_size);
% for i = 1:pic_size(1)
%     for j = 1:pic_size(2)
%        matrix_j(i, j) = double(jamming_pos{i}{j});
%     end
% end
% 
% save("resultIndex", "matrix_j", '-mat');



%% 进行抑制处理
load("matlab\resultIndex.mat"); % 加载训练得到的干扰位置数组
load("matlab\s_mainei.mat"); % 加载原始时频域数组

result = abs(s .* matrix_j);
figure;
subplot(1, 2, 1);
imagesc(linspace(-10,10,size(s,1)),linspace(-10,10,size(s,2)),abs(s));
title('before-suppression');

subplot(1, 2, 2);
imagesc(linspace(-10,10,size(s,1)),linspace(-10,10,size(s,2)),result);
title('after-suppression');

figure;
x = stftmag2sig(abs(s), size(s,1));
xr = stftmag2sig(result, size(result,1));
plot(x);
hold on
plot(xr, '--', LineWidth=2);
hold off
legend('before-suppression', 'after-suppression');