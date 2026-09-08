function []=reorganize()
cvtfinal=evalin('base','cvtfinal_GR');
% cvtfinalx=evalin('base','cvtfinalx');
% cvtfinaly=evalin('base','cvtfinaly');
rescale=nan(2,10000)
for i=1:100
    for j=1:50
       rescale(1,(i-1)*100+j)=i/j;
       rescale(2,(i-1)*100+j)=cvtfinal(i,j);
       % rescalex(1,(i-1)*10+j)=i/j;
       % rescalex(2,(i-1)*10+j)=cvtfinalx(i,j);
       % rescaley(1,(i-1)*10+j)=i/j;
       % rescaley(2,(i-1)*10+j)=cvtfinaly(i,j);
    end
end

assignin('base','rescale',rescale)
% assignin('base','rescalex',rescalex)
% 
% assignin('base','rescaley',rescaley)

figure
%scatter(rescale(1,:),rescale(2,:))
plot(rescale(1,:),rescale(2,:))
xlim([0 5])
end