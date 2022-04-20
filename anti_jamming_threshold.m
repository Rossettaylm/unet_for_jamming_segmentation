%%% 采用设置门限对ISRJ重复转发干扰信号进行抑制

load("matlab\s_mainei.mat"); % 加载原始时频域数组

% threshold = 539.0152;
% threshold = 400;
threshold = 700;

pic_size = size(s);
result = zeros(pic_size);

for i = 1:pic_size(1)
    for j = 1:pic_size(2)
       if abs(s(i, j)) > threshold
            result(i, j) = s(i, j) * 0;
       else 
           result(i, j) = s(i, j);
       end
    end
end
result = abs(result);


%% 作图
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

