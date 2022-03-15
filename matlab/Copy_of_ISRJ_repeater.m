%2021年12月6日
%产生间歇采样转发干扰作为分类数据，干噪比30-60dB之间随机；

%%% 间歇采样循环转发干扰
close all;clear;clc
j=sqrt(-1);
data_num=1;   %干扰样本数
samp_num=5000;%距离窗点数
fs = 100e6; %采样频率
B = 50e6;  %信号带宽
taup = 20e-6; %信号脉宽
N = taup * fs; % 采样点数
t = linspace(taup/2,taup/2*3,N);          %时间序列
k = B / taup;
lfm = exp(1j*pi*k*t.^2);          %LFM信号 复包络

SNR=10; %信噪比dB
echo=zeros(data_num,samp_num,3);     %矩阵大小（500,2000,2）
echo_stft=zeros(data_num,100,247,3);  %矩阵大小（500,200,1000,2）
num_label = 2;
label=zeros(1,data_num)+num_label;                         %标签数据,此干扰标签为0

%% 转发参数设置
% repetion_times_range=[4,3,2];   %重复次数
% period_range=[5e-6, 6.66e-6, 10e-6];    %采样脉冲周期 taup / period = 4 或 2，表示采样次数
% duty_range=[20,25,33.33];  %占空比
duty = 100 / 7; % 占空比为1:7
period = 20e-6 / 2;
repeat_times = 100 / duty;

for m=1:data_num
    %% 目标回波＋噪声
    JNR=20+round(rand(1,1)*20); %干噪比20-40dB
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
%     As=10^(SNR/20);%目标回波幅度
%     Aj=10^(JNR/20);%干扰回波幅度
    As = SNR; Aj = JNR;
    range_tar=1+round(rand(1,1)*2400);
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %噪声+目标回波 目标在距离窗内200点处
    
    %% 采样
    taup_samp_num = taup / period;
    squa=(square((1/period)*2*pi*t, duty)+1)/2;   %生成单极性方波，来做采样
%     squa(400)=0;
    squa1=lfm.*squa;    %采样后的目标回波

    %% 转发
    delay_time=period*(duty*0.01);  %延迟一个采样脉冲时间，即采样后立即转发
    delay_num=ceil(delay_time / (1/fs));  %ceil()为进一法取整，表示一个延迟时间内程序采样点数，计算得在20~50之间
    for i=1:repeat_times %多次重复转发
        %干扰回波幅度×采样后波形
        sp(1+range_tar+i*delay_num : length(lfm)+range_tar+i*delay_num)=sp(1+range_tar+i*delay_num : length(lfm)+range_tar+i*delay_num)+Aj*squa1;
    end
    

    sp=sp/max(sp); %归一化
    sp_abs=abs(sp);

    %%作ISRJ时域波形
%     figure(1)
%     plot(linspace(0,100,length(sp)),sp);
%     set(gca,'FontName','Times New Roman');
%     title("ISRJ")
%     xlabel('Time/μs','FontSize',15);ylabel('Normalized amplitude','FontSize',15)

    %信号实部、虚部分开存入三维张量中
    echo(m,1:5000,1)=real(sp); 
    echo(m,1:5000,2)=imag(sp);
    echo(m,1:5000,3)=sp_abs; 
%     echo(m,1:2000,4)=angle(sp); 

    %%STFT变换
    [S,~,~,~]=spectrogram(sp,32,32-8,512,fs);
    S = imresize(S,[539,682],'nearest');
    S=S/max(max(S));
    S_abs=abs(S);

    %% 作时频图
    h = figure(2);
    ax = axes('Parent', h);
    imagesc(linspace(-10,10,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
    ax.XAxis.Visible = 'off';
    ax.YAxis.Visible = 'off';
    filename = ['repeater', num2str(m), '.jpg'];
    exportgraphics(h, filename);
    


%     set(gca,'FontName','Times New Roman');
%     title("ISRJ | STFT")
%     xlabel('Time/μs','FontSize',15);ylabel('Frequency/MHz','FontSize',15)
%     saveas(gcf, ['repeater', num2str(m)], 'jpg');

    %% 保存实部、虚部、模值
%     echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
%     echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
%     echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs;
%     echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);
 
    
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




