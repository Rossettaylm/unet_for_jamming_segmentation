%2021��12��6��
%������Ъ����ת��������Ϊ�������ݣ������30-60dB֮�������


%%% ��Ъ����ֱ��ת������
close all;clear;clc
j=sqrt(-1);
data_num=1;   %����������
samp_num=6000;%���봰����
fs = 100e6; %����Ƶ��

B = 50e6;  %�źŴ���
taup = 20e-6; %�ź�����
N = taup * fs; % ��������
t = linspace(taup/2,taup*3/2,N);          %ʱ������
k = - B / taup;
lfm = exp(1j*pi*k*t.^2);          %LFM�ź� ������

SNR=10; %�����dB

for m=1:data_num
    
    %% Ŀ��ز�������
    JNR=20+round(rand(1,1)*20); %�����20-40dB
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%��������
    sp=sp/std(sp);
%     As=10^(SNR/20);%Ŀ��ز�����
%     Aj=10^(JNR/20);%���Żز�����
    As = SNR; Aj = JNR;
    range_tar=randi([500, 3000]);  % Ŀ��ز������� 3000 + 2000 + 833 < 6000 �Ϳ��� 833����duty��period����õ�
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %����+Ŀ��ز� Ŀ���ھ��봰��200�㴦
    
  %% ������������
    period= 20e-6 / randi([2, 7]);    %������������ 
    duty=50;  %ռ�ձ�
    repetion_times = 1; % һ�����������ڷ��Ĵ���

    %% ����
%     index1=1+round(rand(1,1));
    period1=period(1);
    duty1=duty(1);
    repetion_times1=repetion_times(1);
    squa=(square((1/period1)*2*pi*t, duty1)+1)/2;   %���ɵ����Է�������������
    squa = move_squa(squa);
    squa1=lfm.*squa;    %�������Ŀ��ز�

    %% ת��
    delay_time=period1*duty1*0.01;  %�ӳ�һ����������ʱ�䣬������������ת��
    delay_num=ceil(delay_time / (1/fs));  %ceil()Ϊ��һ��ȡ������ʾһ���ӳ�ʱ���ڳ�������������������20~50֮��
    for i=1:repetion_times1 %���ת��
        %���Żز����ȡ���������
        sp(1+range_tar+i*delay_num : length(lfm)+range_tar+i*delay_num)=sp(1+range_tar+i*delay_num : length(lfm)+range_tar+i*delay_num)+Aj*squa1;
      
    end

    % ��һ��
    sp = sp / max(sp);


    %% ���ز��źŵ�ʱ����
    figure;
    subplot(2, 2, 1);
    plot(linspace(0, 3, 6000)*taup, real(sp));
    title("���Ե�Ƶ�źŵ�ʱ������")
    xlabel('Time/��s');

    %% �ز��źŵķ�Ƶ����
    subplot(2, 2, 2);
    freq=linspace(-fs,fs,length(sp));
    plot(freq*1e-6,fftshift(abs(fft(sp))));
    xlabel('Frequency in MHz');
    title(' ���Ե�Ƶ�źŵķ�Ƶ����');
    grid on;axis tight;

   %% ��������ѹ������
    st = conj(fliplr(lfm));
    temp = conv(sp(1+range_tar : range_tar+length(lfm)), st);
    sp_pc = temp(ceil(length(temp)/2):end); % 2000:3999 ȡ2000��
    sp_pc = cat(2, sp(1 : range_tar), sp_pc, sp(range_tar+length(lfm)+1:end));

    %% ��sp_pc�źŵĺ�����תΪ����
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

    %% ��sp_pc��ʱ���Լ�Ƶ�������ͼ
    subplot(2, 2, 3);
    plot(linspace(0, 3, 6000)*taup, real(sp_pc));
    title("����ѹ�����ʱ���ź�")
    xlabel('Time/��s');

    subplot(2, 2, 4);
    freq=linspace(-fs,fs,length(sp_pc));
    plot(freq*1e-6,fftshift(abs(fft(sp_pc))));
    xlabel('Frequency in MHz');
    title(' ����ѹ����ķ�Ƶ����');
    grid on;axis tight;

    %% STFT�任
    [S,~,~,~]=spectrogram(sp,32,32-8,512,fs);
    S = imresize(S,[539,682],'nearest');
    S=S/max(max(S));

    [S_pc,~,~,~]=spectrogram(sp_pc,32,32-8,512,fs);
    S_pc = imresize(S_pc,[539,682],'nearest');
    S_pc=S_pc/max(max(S_pc));

    %% ��ʱƵͼ
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
%     xlabel('Time/��s','FontSize',15);ylabel('Frequency/MHz','FontSize',15)
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




