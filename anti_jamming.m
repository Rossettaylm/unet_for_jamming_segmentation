%%% anti-jamming
%%% date: 2022年3月2日
%% 通过python函数获取干扰识别后的干扰信号位置数据
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

unet = py.unet.Unet();
pos = py.unet.Unet.get_jamming_pos(unet, "./img/direct.jpg");
length(pos)
length(pos{1})