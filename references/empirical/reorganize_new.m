function []=reorganize()
cvtfinal=evalin('base','cvtfinal_R');
% cvtfinalx=evalin('base','cvtfinalx');
% cvtfinaly=evalin('base','cvtfinaly');
%rescale=nan(2,10000)
for i=1:size(cvtfinal,1)
    for j=1:50
       % rescale(1,(i-1)*100+j)=i/j;
       % rescale(2,(i-1)*100+j)=cvtfinal(i,j);
      cvt_rescale(i,j)=cvtfinal(i,j);
    end
end


FrameT=0.05;
labelt=(0:5:50)*FrameT;
figure(5)
   for j=10:50
   plot((1:751)/j, cvtfinal(:,j),LineWidth=2);
    hold on ;%normally, no need to loop
   end 

colororder (turbo(size(cvtfinal,2)/2));
 colormap (turbo(size(cvtfinal,2)/2)); 
  colororder (turbo(50));
 colormap (turbo(50)); 
 colorbar('TickLabels',labelt);
xlim([0 2.5])

assignin('base','rescale',rescale)
% assignin('base','rescalex',rescalex)
% 
% assignin('base','rescaley',rescaley)

figure(3)
%scatter(rescale(1,:),rescale(2,:))
plot(rescale(1,:),rescale(2,:))
xlim([0 5])
end