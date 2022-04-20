%2021年12月6日
%产生间歇采样转发干扰作为分类数据，干噪比30-60dB之间随机；


%%% 间歇采样循环转发干扰
close all;clear;clc
j=sqrt(-1);
data_num=1;   %干扰样本数
samp_num=6000;%距离窗点数
fs = 100e6; %采样频率

B = 50e6;  %信号带宽
taup = 20e-6; %信号脉宽
N = taup * fs; % 采样点数
t = linspace(taup/2,taup*3/2,N);          %时间序列
k = - B / taup;
lfm = exp(1j*pi*k*t.^2);          %LFM信号 复包络

SNR=10; %信噪比dB

for m=1:data_num
    
    %% 目标回波＋噪声
    JNR=20+round(rand(1,1)*20); %干噪比20-40dB
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
%     As=10^(SNR/20);%目标回波幅度
%     Aj=10^(JNR/20);%干扰回波幅度
    As = SNR; Aj = JNR;
    range_tar=randi([500, 3000]);  % 目标回波最大点数 3000 + 2000 + 833 < 6000 就可以 833根据duty和period计算得到
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %噪声+目标回波 目标在距离窗内200点处
    

 %% 采样参数设置

    section_num_list = [5, 9, 14];
    loop_num = randi([2, 4]); % 循环转发次数
    section_num = section_num_list(loop_num - 1);

    section = taup / section_num; % 切片的长度
    section_samp_num = N / (taup / section); % 切片的采样点数
    section_samp_num = int16(section_samp_num);

    DRFM = zeros(1, loop_num*section_samp_num); % 将每次采样后的信号按倒序存入转发列表

    %% 转发
    for i=1:loop_num
        a = (i * (i+1) / 2 - 1); % 截获切片开始的坐标0、2、5、9...
        current_samp = sp(1+range_tar+a*section_samp_num : range_tar+(a+1)*section_samp_num);
        DRFM(1+(loop_num-i)*section_samp_num : (loop_num-i+1)*section_samp_num) = current_samp;
%         从采样切片的下一个切片开始转发
        sp(1+range_tar+(a+1)*section_samp_num : range_tar+(a+1+i)*section_samp_num) =  Aj * DRFM(1+(loop_num-i)*section_samp_num : (loop_num)*section_samp_num) +  sp(1+range_tar+(a+1)*section_samp_num : range_tar+(a+1+i)*section_samp_num);
    end
    

    sp=sp/max(sp); %归一化


    %% 作回波信号的时域波形
    figure;
    subplot(2, 2, 1);
    plot(linspace(0, 3, 6000)*taup, real(sp));
    title("线性调频信号的时域特性")
    xlabel('Time/μs');

    %% 回波信号的幅频特性
    subplot(2, 2, 3);
    freq=linspace(-fs,fs,length(sp));
    plot(freq*1e-6,fftshift(abs(fft(sp))));
    xlabel('Frequency in MHz');
    title(' 线性调频信号的幅频特性');
    grid on;axis tight;

   %% 进行脉冲压缩处理
    st = conj(fliplr(lfm));
    temp = conv(sp(1+range_tar : range_tar+length(lfm)), st);
    sp_pc = temp(ceil(length(temp)/2):end); % 2000:3999 取2000点
    sp_pc = cat(2, sp(1 : range_tar), sp_pc, sp(range_tar+length(lfm)+1:end));

    %% 将sp_pc信号的横坐标转为距离
%     c = 3.0e8;
%     t_to_distance = c / (2 * fs);
%     
%     temp = zeros(1,size(sp_pc,2));
%     index_end = floor(6000 / t_to_distance);
%     for index=1:index_end
%         R_index = floor(index * t_to_distance);
%         temp(R_index) = sp(index);
%     end
%     sp_pc = temp;

    %% 对sp_pc的时域以及频域进行作图
    subplot(2, 2, 2);
    plot(linspace(0, 3, 6000)*taup, real(sp_pc));
    title("脉冲压缩后的时域信号")
    xlabel('Time/μs');

    subplot(2, 2, 4);
    freq=linspace(-fs,fs,length(sp_pc));
    plot(freq*1e-6,fftshift(abs(fft(sp_pc))));
    xlabel('Frequency in MHz');
    title(' 脉冲压缩后的幅频特性');
    grid on;axis tight;

    %% STFT变换
    [S,~,~,~]=spectrogram(sp,32,32-8,512,fs);
    S = imresize(S,[539,682],'nearest');
    S=S/max(max(S));

    [S_pc,~,~,~]=spectrogram(sp_pc,32,32-8,512,fs);
    S_pc = imresize(S_pc,[539,682],'nearest');
    S_pc=S_pc/max(max(S_pc));

    %% 作时频图
    h = figure;
    ax = axes('Parent', h);
    imagesc(linspace(-10,10,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
    ax.XAxis.Visible = 'off';
    ax.YAxis.Visible = 'off';
%     filename = ['repeater', num2str(m), '.jpg'];
%     exportgraphics(h, filename);

    h = figure;
    ax = axes('Parent', h);
    imagesc(linspace(-10,10,size(S_pc,1)),linspace(-10,10,size(S_pc,2)),abs(S_pc));
    ax.XAxis.Visible = 'off';
    ax.YAxis.Visible = 'off';


%     set(gca,'FontName','Times New Roman');
%     title("ISRJ | STFT")
%     xlabel('Time/μs','FontSize',15);ylabel('Frequency/MHz','FontSize',15)
%     saveas(gcf, ['repeater', num2str(m)], 'jpg');

 
    
end

% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\echo.mat' ,'echo')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\echo_stft.mat' ,'echo_stft')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\label.mat' ,'label')

% t_data=load('D:\CodeSpace\active_jamming_recognition\data\t_data.mat').t_data;
% tf_data=load('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat').tf_data;
% gt_label=load('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat').gt_label;
% % 
% t_data(1+500*(num_label):500*(num_label+1),:,:)=echo; 
% tf_data(1+500*(num_label):500*(num_label+1),:,:,:)=echo_stft; 
% gt_label(1,1+500*(num_label):500*(num_label+1))=label;
% % 
% save('D:\CodeSpace\active_jamming_recognition\data\t_data.mat','t_data')
% save('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat','tf_data')
% save('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat','gt_label')


% figure(1)
% plot(1:400,squa)
% 
% figure(2)
% plot(1:400,squa1)
% figure(3)
% plot(1:2000,sp)
% figure(4)
% [S,~,~,~]=spectrogram(sp,32*2,32*2-1,200,20e6);
% 
% S=S/max(max(S));
% imagesc(1:size(S,1),1:size(S,2),abs(S))




