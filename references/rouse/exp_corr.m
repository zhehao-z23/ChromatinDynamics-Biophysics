figure
trajsize=5;
curve=[1 5 10 50 100];
labelt=curve*FrameT;
plot(matrix(:,curve), cvtfinal_GR(:,curve),LineWidth=2); %normally, no need to loop
 % colororder (turbo(size(cvtfinal_GR,2)/2));
 % colormap (turbo(size(cvtfinal_GR,2)/2)); 
  colororder (turbo(trajsize));
 colormap (turbo(trajsize)); 
%  colorbar('TickLabels',labelt,FontSize=15);
 colorbar('Ticks', [1,2,3,4,5]*0.2-0.1,'TickLabels',labelt,FontSize=15);
 xlim([0 2.5])

 % cvtfinal_GR is the y value for exp, vvcf is the y value for theor,
 % matrix is the x value for exp, 1,5 10,50,100 column ( first 3 13 26 126 251 rows is < 2.5) , time is the x
 % value for theo, but it need index from exp,