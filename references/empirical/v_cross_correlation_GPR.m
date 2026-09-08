function []= v_cross_correlation_GPR()
% author: Yanyu Zhu created in 12/8/2016
% the code calculate the velocity cross-correlation  between the with the
% input of Dfin and Dfiny

prompt = {'time betwee frames:',...
               'length of pixel'}; 
        u_name = 'Input parameters';
        numlines = 1;
        defaultanswer = {'1','.001'};
        options.Resize = 'on';
        options.WindowStyle = 'normal';
        options.Interpreter = 'tex';
        user_var = inputdlg(prompt,u_name,numlines,defaultanswer,options);
        FrameT= str2double(user_var{1});
        pixl= str2double(user_var{2});

     % reading the Dfin and Dfiny Matrices from the workspace  
     %    traj=6;
        % Dfin_P_new=evalin('base','Dfin_P(:,traj)');
        % Dfiny_P_new=evalin('base','Dfiny_P(:,traj)');
        % 
       %  Dfin_G_new=evalin('base','Dfin_G(:,:)');
       %  Dfiny_G_new=evalin('base','Dfiny_G(:,:)');
        % 
        % Dfin_R_new=evalin('base','Dfin_R(:,traj)');
        % Dfiny_R_new=evalin('base','Dfiny_R(:,traj)');

         Dfin_P_new=evalin('base','Dfin_P(:,:)');
        Dfiny_P_new=evalin('base','Dfiny_P(:,:)');

        Dfin_G_new=evalin('base','Dfin_R(:,:)');
        Dfiny_G_new=evalin('base','Dfiny_R(:,:)');
        % 
        % Dfin_R_new=evalin('base','Dfin_R(:,:)');
        % Dfiny_R_new=evalin('base','Dfiny_R(:,:)');
     
        % figure(111)
        % plot(Dfin_G_new,-Dfiny_G_new,'g');
        % hold on;
        % plot(Dfin_R_new,-Dfiny_R_new,'r');
        % hold on;
        % plot(Dfin_P_new,-Dfiny_P_new,'m');
        %%
        % Dfin_P_cor= Dfin_P_new-(Dfin_P_new+Dfin_G_new+Dfin_R_new)/3;
        % Dfiny_P_cor= Dfiny_P_new-(Dfiny_P_new+Dfiny_G_new+Dfiny_R_new)/3;
        % Dfin_G_cor= Dfin_G_new-(Dfin_P_new+Dfin_G_new+Dfin_R_new)/3;
        % Dfiny_G_cor= Dfiny_G_new-(Dfiny_P_new+Dfiny_G_new+Dfiny_R_new)/3;
        % Dfin_R_cor= Dfin_R_new-(Dfin_P_new+Dfin_G_new+Dfin_R_new)/3;
        % Dfiny_R_cor= Dfiny_R_new-(Dfiny_P_new+Dfiny_G_new+Dfiny_R_new)/3;
        % 
        % Dfin_G_new=Dfin_G_cor;
        % Dfiny_G_new=Dfiny_G_cor;
        % Dfin_P_new=Dfin_P_cor;
        % Dfiny_P_new=Dfiny_P_cor;
        % Dfin_R_new=Dfin_R_cor;
        % Dfiny_R_new=Dfiny_R_cor;
        % figure(112)
        % plot(Dfin_G_new,-Dfiny_G_new,'g',LineWidth=2);
        % hold on
        % plot(Dfin_P_new,-Dfiny_P_new,'m',LineWidth=2);
        % hold on
        % plot(Dfin_R_new,-Dfiny_R_new,'r',LineWidth=2);
        %%
        % Dfin_R=evalin('base','Dfin_rep(:,:)');
        % Dfiny_R=evalin('base','Dfiny_rep(:,:)');
        % 
        % Dfin_G=evalin('base','Dfin_nonrep(:,:)');
        % Dfiny_G=evalin('base','Dfiny_nonrep(:,:)');
%%
trajno=size(Dfin_P_new,2);
CutoffLen=size(Dfin_P_new,1);

delta=FrameT;
cvt_R=zeros(CutoffLen/2,CutoffLen-1);
cvtx_R=zeros(CutoffLen/2,CutoffLen-1);
cvty_R=zeros(CutoffLen/2,CutoffLen-1);
cvt_G=zeros(CutoffLen/2,CutoffLen-1);
cvtx_G=zeros(CutoffLen/2,CutoffLen-1);
cvty_G=zeros(CutoffLen/2,CutoffLen-1);

 for I=0:CutoffLen/2 % possible ways to have lag time
   for d=1:CutoffLen/2    %  calculate velocity delta
     vr_R = zeros(CutoffLen-d,trajno);
     vy_R = zeros(CutoffLen-d,trajno);
     vx_R = zeros(CutoffLen-d,trajno);
     vr_G = zeros(CutoffLen-d,trajno);
     vy_G = zeros(CutoffLen-d,trajno);
     vx_G = zeros(CutoffLen-d,trajno);
     cv_R=zeros(CutoffLen-d-I,trajno);    % to calculate auto-correlation for normalize
     cvx_R=zeros(CutoffLen-d-I,trajno);
     cvy_R=zeros(CutoffLen-d-I,trajno);
     cv=zeros(CutoffLen-d-I,trajno);    % calculate cross-correlation 
     cvx=zeros(CutoffLen-d-I,trajno);
     cvy=zeros(CutoffLen-d-I,trajno);

    for i=1:trajno
        for j=d+1:CutoffLen
            if Dfiny_P_new(j,i)~=0
                vx_R(j-d,i)=(Dfin_P_new(j,i)-Dfin_P_new((j-d),i))/(delta*d);
                vy_R(j-d,i)=(Dfiny_P_new(j,i)-Dfiny_P_new((j-d),i))/(delta*d);
             %   vr(j-I,i)=vx(j-I,i)+vy(j-I,i);
            end
            if Dfiny_G_new(j,i)~=0
                vx_G(j-d,i)=(Dfin_G_new(j,i)-Dfin_G_new((j-d),i))/(delta*d);
                vy_G(j-d,i)=(Dfiny_G_new(j,i)-Dfiny_G_new((j-d),i))/(delta*d);
             %   vr(j-I,i)=vx(j-I,i)+vy(j-I,i);
            end

        end
    end  
    for j=1+I:CutoffLen-d
        for i=1:trajno
            cv_R(j-I,i)=vx_R(j,i)*vx_R(j-I,i)+vy_R(j,i)*vy_R(j-I,i);  % I is like tao,lag time
            cvx_R(j-I,i)=vx_R(j,i)*vx_R(j-I,i);
            cvy_R(j-I,i)=vy_R(j,i)*vy_R(j-I,i);
            cv(j-I,i)=vx_R(j,i)*vx_G(j-I,i)+vy_R(j,i)*vy_G(j-I,i);  % I is like tao,lag time
            cvx(j-I,i)=vx_R(j,i)*vx_G(j-I,i);
            cvy(j-I,i)=vy_R(j,i)*vy_G(j-I,i);

        end
    end
    cvt_R(d,I+1)=nanmean(nanmean(cv_R));
    cvtx_R(d,I+1)=nanmean(nanmean(cvx_R));
    cvty_R(d,I+1)=nanmean(nanmean(cvy_R));
    cvt(d,I+1)=nanmean(nanmean(cv));
    cvtx(d,I+1)=nanmean(nanmean(cvx));
    cvty(d,I+1)=nanmean(nanmean(cvy));


   end
end
% to get the first point as zero assigning a new matrix

cvttemp_R=zeros(CutoffLen/2,CutoffLen-1);
cvttempx_R=zeros(CutoffLen/2,CutoffLen-1);
cvttempy_R=zeros(CutoffLen/2,CutoffLen-1);

cvttemp=zeros(CutoffLen/2,CutoffLen-1);
cvttempx=zeros(CutoffLen/2,CutoffLen-1);
cvttempy=zeros(CutoffLen/2,CutoffLen-1);

for j=1:CutoffLen/2
    for i=1:CutoffLen/2

     cvttemp_R(i,j)=cvt_R(i,j+1)/cvt_R(i,1);  % auto-correlation
     cvttempx_R(i,j)=cvtx_R(i,j+1)/cvtx_R(i,1);
     cvttempy_R(i,j)=cvty_R(i,j+1)/cvty_R(i,1);

     cvttemp(i,j)=cvt(i,j)/cvt_R(i,1);  % cross-correlation 
     cvttempx(i,j)=cvtx(i,j)/cvtx_R(i,1);
     cvttempy(i,j)=cvty(i,j)/cvty_R(i,1);
    end
end
cvtfinalx_GR=zeros(CutoffLen/2+1,CutoffLen/2);
cvtfinal_GR=zeros(CutoffLen/2+1,CutoffLen/2);
cvtfinaly_GR=zeros(CutoffLen/2+1,CutoffLen/2);

cvtfinalx_R=zeros(CutoffLen/2+1,CutoffLen/2);
cvtfinal_R=zeros(CutoffLen/2+1,CutoffLen/2);
cvtfinaly_R=zeros(CutoffLen/2+1,CutoffLen/2);
for i=1:CutoffLen/2
    for j=1:CutoffLen/2
      cvtfinal_R(i+1,j)=cvttemp_R(j,i);
      cvtfinalx_R(i+1,j)=cvttempx_R(j,i);
      cvtfinaly_R(i+1,j)=cvttempy_R(j,i);
      cvtfinal_R(1,j)=1;
      cvtfinalx_R(1,j)=1;
      cvtfinaly_R(1,j)=1;

      cvtfinal_GR(i,j)=cvttemp(j,i);
      cvtfinalx_GR(i,j)=cvttempx(j,i);
      cvtfinaly_GR(i,j)=cvttempy(j,i);
   
    end
end
%%
assignin('base','cvtfinal_R',cvtfinal_R)
assignin('base','cvtfinalx_R',cvtfinalx_R)
assignin('base','cvtfinaly_R',cvtfinaly_R)

assignin('base','cvtfinal_GR',cvtfinal_GR)
assignin('base','cvtfinalx_GR',cvtfinalx_GR)
assignin('base','cvtfinaly_GR',cvtfinaly_GR)
%%
len=size(Dfin_G_new,1);
%len=size(Dfin_G,1);
FrameT=1;
labelt=(0:1:10)*FrameT;
%labelt=(0:5:50)*FrameT;
matrix=nan(len/2+1,len/2+1);
for i= 1: len/2+1
        matrix(i,:)= (0:len/2)/i;
end
matrix=matrix';
%%
% trajsize=10;
% figure
% plot((1:len/2+1)*FrameT, cvtfinal_R(:,1:trajsize),LineWidth=2); %normally, no need to loop
%  colororder (turbo(size(cvtfinal_R,2)/2));
%  colormap (turbo(size(cvtfinal_R,2)/2)); 
%   colororder (turbo(trajsize));
%  colormap (turbo(trajsize)); 
%  colorbar('TickLabels',labelt);
%xlim([0 5])
%%
% figure
% plot(matrix(:,1:50), cvtfinal_R(:,1:50),LineWidth=2); %normally, no need to loop
%  colororder (turbo(size(cvtfinal_R,2)/2));
%  colormap (turbo(size(cvtfinal_R,2)/2)); 
%   colororder (turbo(50));
%  colormap (turbo(50)); 
%  colorbar('TickLabels',labelt);
% xlim([0 5])
%%
% figure
% trajsize=10;
% plot((1:len/2+1)*FrameT, cvtfinal_GR(:,1:10),LineWidth=2); %normally, no need to loop
%  % colororder (turbo(size(cvtfinal_GR,2)/2));
%  % colormap (turbo(size(cvtfinal_GR,2)/2)); 
%   colororder (turbo(trajsize));
%  colormap (turbo(trajsize)); 
%  colorbar('TickLabels',labelt,FontSize=15);
% %xlim([0 2.5])
%%
figure
trajsize=4;
traj=[1 2 5 10];
plot(matrix(:,traj), cvtfinal_GR(:,traj),LineWidth=4); %normally, no need to loop
 % colororder (turbo(size(cvtfinal_GR,2)/2));
 % colormap (turbo(size(cvtfinal_GR,2)/2)); 
 %  colororder (turbo(trajsize));
 % colormap (turbo(trajsize)); 
 colororder (lines(trajsize));
 colormap (lines(trajsize)); 
% colorbar('TickLabels',labelt,FontSize=15);
  fontsize(35,"points");
for j=1:trajsize
    legend_text{j}=strcat('\delta=',num2str(traj(j)*1), 's');
end

 xlim([0 2.5])
 ylim([-.5 1])
set(gca,'FontSize',35)  
set(gca,'linewidth',4)  
set(gca,'TickDir','out');
set(gca,'TickLength', [0.02 0.02])
legend(legend_text)
xlabel('\tau/\delta','FontSize',35)
ylabel('C_{vv}^{(\delta)}(\tau,\Deltan)/C_{vv}^{(\delta)}(0,0)','FontSize',35)
 %%
figure
trajsize=10;
curve=1:1:10;
labelt=curve*FrameT;
plot(matrix(:,curve), cvtfinal_GR(:,curve),LineWidth=2); %normally, no need to loop
 % colororder (turbo(size(cvtfinal_GR,2)/2));
 % colormap (turbo(size(cvtfinal_GR,2)/2)); 
  colororder (turbo(trajsize));
 colormap (turbo(trajsize)); 
%  colorbar('TickLabels',labelt,FontSize=15);
 colorbar('Ticks', 0.1:0.1:1,'TickLabels',labelt,FontSize=15);
 xlim([0 2.5])
  ylim([-.5 1])
  fontsize(35,"points");

  xlabel('\tau/\delta','FontSize',35)
ylabel('C_{vv}^{(\delta)}(\tau,\Deltan)/C_{vv}^{(\delta)}(0,0)','FontSize',35)
 % %%
 % figure
 % plot((1:1:size(cvtfinal_GR,2))*FrameT, cvtfinal_GR(1,:),LineWidth=2);
 % % xlim([0 5])
 %  xlabel('delta(s)');
 % ylabel('CVV(t=0)');
 % fontsize(35,"points");
%%
FrameT=1;  
figure
 plot(log((1:1:size(cvtfinal_GR,2))*FrameT), log(cvtfinal_GR(1,:)),LineWidth=2);
 % xlim(log([0 5]))
  xlabel('log(delta(s))');
 ylabel('log(CVV(t=0))');
 x=log((1:1:size(cvtfinal_GR,2))*FrameT);
 y=log(cvtfinal_GR(1,:));
 fontsize(35,"points");
 %%
% xlabel('tao (s)');
% ylabel('cv(tao)/cv(0) (nm)');
% fontsize(30,"points");


% figure(4)
% for kk=1:50
%    colormap(hsv)
%     plot(1:101, cvtfinal(:,kk));
%     colororder (turbo(size(cvtfinal,2)));
%     colorbar;
%    hold on;
%    xlim([0 75])
% end 

% assignin('base','cvt',cvt_R)
% assignin('base','cvttemp',cvttemp_R)


x=(0:1:CutoffLen-1)*FrameT;
assignin('base','time',x)
assignin('base','matrix',matrix);

end