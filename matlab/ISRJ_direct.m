%2021��12��7��
%������Ъ����ת��������Ϊ�������ݣ������30-60dB֮�������

%% ��Ъ����ֱ��ת������
close all;clear;clc
j=sqrt(-1);
data_num=1;   %����������
samp_num=5000;%���봰����
fs = 100e6; %����Ƶ��
B = 50e6;  %�źŴ���
taup = 20e-6; %�ź�����
t = linspace(-taup/2,taup/2,taup*fs);          %ʱ������
k = B / taup;
lfm = exp(1j*pi*k*t.^2);          %LFM�ź� ������

SNR=10; %�����dB


for m=1:data_num

    period= 20e-6 / randi([2, 7]);    %������������ 
    duty=50;  %ռ�ձ�
    repetion_times = 1; % һ�����������ڷ��Ĵ���

    %% Ŀ��ز�������
    JNR=20+round(rand(1,1)*20); %�����20-40dB
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%��������
    sp=sp/std(sp);
%     As=10^(SNR/20);%Ŀ��ز�����
%     Aj=10^(JNR/20);%���Żز�����
    As = SNR; Aj = JNR;
    range_tar=1+round(rand(1,1)*2400);
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %����+Ŀ��ز� Ŀ���ھ��봰��200�㴦
    
    %% ����
%     index1=1+round(rand(1,1));
    period1=period(1);
    duty1=duty(1);
    repetion_times1=repetion_times(1);
    squa=(square((1/period1)*2*pi*t, duty1)+1)/2;   %���ɵ����Է�������������
    squa(400)=0;
    squa1=lfm.*squa;    %�������Ŀ��ز�

    %% ת��
    delay_time=period1*duty1*0.01;  %�ӳ�һ����������ʱ�䣬������������ת��
    delay_num=ceil(delay_time / (1/fs));  %ceil()Ϊ��һ��ȡ������ʾһ���ӳ�ʱ���ڳ�������������������20~50֮��
    for i=1:repetion_times1 %���ת��
        %���Żز����ȡ���������
        sp(1+range_tar+i*delay_num : length(lfm)+range_tar+i*delay_num)=sp(1+range_tar+i*delay_num : length(lfm)+range_tar+i*delay_num)+Aj*squa1;
      
    end
    

    sp=sp/max(sp); %��һ��
    sp_abs=abs(sp);

    %% ��ISRJʱ����
%     figure(1)
%     plot(linspace(0,100,2000),sp);
%     set(gca,'FontName','Times New Roman');
%     title("ISRJ_direct")
%     xlabel('Time/��s','FontSize',15);ylabel('Normalized amplitude','FontSize',15)

    figure(2);
    freq=linspace(-fs/2,fs/2,taup*fs);
    plot(freq*1e-6,fftshift(abs(fft(lfm))));
    xlabel('Frequency in MHz');
    title(' ���Ե�Ƶ�źŵķ�Ƶ����');
    grid on;axis tight;

    % �ź�ʵ�����鲿�ֿ�������ά������
%     echo(m,1:5000,1)=real(sp); 
%     echo(m,1:5000,2)=imag(sp);
%     echo(m,1:5000,3)=sp_abs; 
%     echo(m,1:10000,4)=angle(sp); 

    %% STFT�任
    S = stft(sp, fs);
%     [S,~,~,~]=spectrogram(sp,32,32-8,512,fs);
    S = imresize(S,[539,682],'nearest');
    S=S/max(max(S));
    S_abs=abs(S);

    %% ��ʱƵͼ
    h = figure(1);
    ax = axes('Parent', h);
    imagesc(linspace(-10,10,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
    ax.XAxis.Visible = 'off';
    ax.YAxis.Visible = 'off';
%     filename = ['direct', num2str(m), '.jpg'];
%     exportgraphics(h, filename);

%     set(gca,'FontName','Times New Roman');
%     title("ISRJ_direct | STFT")
%     xlabel('Time/��s','FontSize',15);ylabel('Frequency/MHz','FontSize',15)
%     saveas(gcf, ['direct', num2str(m)], 'jpg');


    %% ����ʵ�����鲿��ģֵ
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




