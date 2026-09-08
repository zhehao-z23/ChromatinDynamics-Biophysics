
% author: Yanyu Zhu created in 12/8/2016
% the code calculate the velocity cross-correlation  between the with the
% input of Dfin and Dfiny

prompt = {'time betwee frames:',...
               'length of pixel'}; 
        u_name = 'Input parameters';
        numlines = 1;
        defaultanswer = {'0.05','.001'};
        options.Resize = 'on';
        options.WindowStyle = 'normal';
        options.Interpreter = 'tex';
        user_var = inputdlg(prompt,u_name,numlines,defaultanswer,options);
        FrameT= str2double(user_var{1});
        pixl= str2double(user_var{2});

     % reading the Dfin and Dfiny Matrices from the workspace  
     
FrameT=0.05;
d=4090;
col=1:30;
  Dfin_R=evalin('base','Dfin_rep(1:d,col)');
  Dfiny_R=evalin('base','Dfiny_rep(1:d,col)');
  Dfinz_R=evalin('base','Dfinz_rep(1:d,col)');

  Dfin_G=evalin('base','Dfin_nonrep(1:d,col)');
  Dfiny_G=evalin('base','Dfiny_nonrep(1:d,col)');
  Dfinz_G=evalin('base','Dfinz_nonrep(1:d,col)');


trajno=size(Dfin_R,2);
CutoffLen=size(Dfin_R,1);
deltarange=500;
delta=FrameT;
cvt_R=zeros(deltarange,CutoffLen-1);
cvtx_R=zeros(deltarange,CutoffLen-1);
cvty_R=zeros(deltarange,CutoffLen-1);
cvtz_R=zeros(deltarange,CutoffLen-1);
cvt_G=zeros(deltarange,CutoffLen-1);
cvtx_G=zeros(deltarange,CutoffLen-1);
cvty_G=zeros(deltarange,CutoffLen-1);  
cvtz_G=zeros(deltarange,CutoffLen-1);

cvt_R_traj=zeros(deltarange,CutoffLen-1,trajno);
cvtx_R_traj=zeros(deltarange,CutoffLen-1,trajno);
cvty_R_traj=zeros(deltarange,CutoffLen-1,trajno);
cvtz_R_traj=zeros(deltarange,CutoffLen-1,trajno);
cvt_G_traj=zeros(deltarange,CutoffLen-1,trajno);
cvtx_G_traj=zeros(deltarange,CutoffLen-1,trajno);
cvty_G_traj=zeros(deltarange,CutoffLen-1,trajno);  
cvtz_G_traj=zeros(deltarange,CutoffLen-1,trajno);

cvt_traj=zeros(deltarange,CutoffLen-1,trajno);  % for corss-correlation
cvtx_traj=zeros(deltarange,CutoffLen-1,trajno);
cvty_traj=zeros(deltarange,CutoffLen-1,trajno);
cvtz_traj=zeros(deltarange,CutoffLen-1,trajno);

%msdtraj = nan(trajno,deltarange-1) ;                % the msd of each single traj

 for I=0:deltarange % possible ways to have lag time
   for d=1:deltarange    %  calculate velocity delta
     vr_R = zeros(CutoffLen-d,trajno);
     vy_R = zeros(CutoffLen-d,trajno);
     vx_R = zeros(CutoffLen-d,trajno);
     vz_R = zeros(CutoffLen-d,trajno);
     vr_G = zeros(CutoffLen-d,trajno);
     vy_G = zeros(CutoffLen-d,trajno);
     vx_G = zeros(CutoffLen-d,trajno);
     vz_G = zeros(CutoffLen-d,trajno);
     cv_R=zeros(CutoffLen-d-I,trajno);    % to calculate auto-correlation for normalize
     cvx_R=zeros(CutoffLen-d-I,trajno);
     cvy_R=zeros(CutoffLen-d-I,trajno);
     cvz_R=zeros(CutoffLen-d-I,trajno);
     cv=zeros(CutoffLen-d-I,trajno);    % calculate cross-correlation 
     cvx=zeros(CutoffLen-d-I,trajno);
     cvy=zeros(CutoffLen-d-I,trajno);
     cvz=zeros(CutoffLen-d-I,trajno);
    for i=1:trajno
        for j=d+1:CutoffLen
            if Dfiny_R(j,i)~=0
                vx_R(j-d,i)=(Dfin_R(j,i)-Dfin_R((j-d),i))/(delta*d);
                vy_R(j-d,i)=(Dfiny_R(j,i)-Dfiny_R((j-d),i))/(delta*d);
                vz_R(j-d,i)=(Dfinz_R(j,i)-Dfinz_R((j-d),i))/(delta*d);
             %   vr(j-I,i)=vx(j-I,i)+vy(j-I,i);
            end
            if Dfiny_G(j,i)~=0
                vx_G(j-d,i)=(Dfin_G(j,i)-Dfin_G((j-d),i))/(delta*d);
                vy_G(j-d,i)=(Dfiny_G(j,i)-Dfiny_G((j-d),i))/(delta*d);
                vz_G(j-d,i)=(Dfinz_G(j,i)-Dfinz_G((j-d),i))/(delta*d);
             %   vr(j-I,i)=vx(j-I,i)+vy(j-I,i);
            end

        end
    end  
    for j=1+I:CutoffLen-d
        for i=1:trajno
            cv_R(j-I,i)=vx_R(j,i)*vx_R(j-I,i)+vy_R(j,i)*vy_R(j-I,i)+vz_R(j,i)*vz_R(j-I,i);  % I is like tao,lag time
            cvx_R(j-I,i)=vx_R(j,i)*vx_R(j-I,i);
            cvy_R(j-I,i)=vy_R(j,i)*vy_R(j-I,i);
            cvz_R(j-I,i)=vz_R(j,i)*vz_R(j-I,i);
            cv(j-I,i)=vx_R(j,i)*vx_G(j-I,i)+vy_R(j,i)*vy_G(j-I,i)+vz_R(j,i)*vz_G(j-I,i);  % I is like tao,lag time
            cvx(j-I,i)=vx_R(j,i)*vx_G(j-I,i);
            cvy(j-I,i)=vy_R(j,i)*vy_G(j-I,i);
            cvz(j-I,i)=vz_R(j,i)*vz_G(j-I,i);

        end
    end
    cvt_R(d,I+1)=nanmean(nanmean(cv_R));
    cvtx_R(d,I+1)=nanmean(nanmean(cvx_R));
    cvty_R(d,I+1)=nanmean(nanmean(cvy_R));
    cvtz_R(d,I+1)=nanmean(nanmean(cvz_R));
    cvt(d,I+1)=nanmean(nanmean(cv));
    cvtx(d,I+1)=nanmean(nanmean(cvx));
    cvty(d,I+1)=nanmean(nanmean(cvy));
    cvtz(d,I+1)=nanmean(nanmean(cvz));

    cvt_R_traj(d,I+1,1:trajno)=nanmean(cv_R);
    cvtx_R_traj(d,I+1,1:trajno)=nanmean(cvx_R);
    cvty_R_traj(d,I+1,1:trajno)=nanmean(cvy_R);
    cvtz_R_traj(d,I+1,1:trajno)=nanmean(cvz_R);
    cvt_traj(d,I+1,1:trajno)=nanmean(cv);
    cvtx_traj(d,I+1,1:trajno)=nanmean(cvx);
    cvty_traj(d,I+1,1:trajno)=nanmean(cvy);
    cvtz_traj(d,I+1,1:trajno)=nanmean(cvz);

   end
end
% to get the first point as zero assigning a new matrix

cvttemp_R=zeros(deltarange,CutoffLen-1);
cvttempx_R=zeros(deltarange,CutoffLen-1);
cvttempy_R=zeros(deltarange,CutoffLen-1);
cvttempz_R=zeros(deltarange,CutoffLen-1);

cvttemp=zeros(deltarange,CutoffLen-1);
cvttempx=zeros(deltarange,CutoffLen-1);
cvttempy=zeros(deltarange,CutoffLen-1);
cvttempz=zeros(deltarange,CutoffLen-1);

cvttemp_R_traj=zeros(deltarange,CutoffLen-1,trajno);
cvttempx_R_traj=zeros(deltarange,CutoffLen-1,trajno);
cvttempy_R_traj=zeros(deltarange,CutoffLen-1,trajno);
cvttempz_R_traj=zeros(deltarange,CutoffLen-1,trajno);

cvttemp_traj=zeros(deltarange,CutoffLen-1,trajno);
cvttempx_traj=zeros(deltarange,CutoffLen-1,trajno);
cvttempy_traj=zeros(deltarange,CutoffLen-1,trajno);
cvttempz_traj=zeros(deltarange,CutoffLen-1,trajno);


for j=1:deltarange
    for i=1:deltarange

     cvttemp_R(i,j)=cvt_R(i,j+1)/cvt_R(i,1);  % auto-correlation
     cvttempx_R(i,j)=cvtx_R(i,j+1)/cvtx_R(i,1);
     cvttempy_R(i,j)=cvty_R(i,j+1)/cvty_R(i,1);
     cvttempz_R(i,j)=cvtz_R(i,j+1)/cvtz_R(i,1);
     cvttemp(i,j)=cvt(i,j)/cvt_R(i,1);  % cross-correlation 
     cvttempx(i,j)=cvtx(i,j)/cvtx_R(i,1);
     cvttempy(i,j)=cvty(i,j)/cvty_R(i,1);
     cvttempz(i,j)=cvtz(i,j)/cvtz_R(i,1);
   
    end
end


for j=1:deltarange
    for i=1:deltarange
       for kk=1:trajno
     cvttemp_R_traj(i,j,kk)=cvt_R_traj(i,j+1,kk)/cvt_R_traj(i,1,kk);  % auto-correlation
     % cvttempx_R_traj(i,j,trajno)=cvtx_R(i,j+1,trajno)/cvtx_R(i,1,trajno);
     % cvttempy_R_traj(i,j,trajno)=cvty_R(i,j+1,trajno)/cvty_R(i,1,trajno);
     % cvttempz_R_traj(i,j,trajno)=cvtz_R(i,j+1,trajno)/cvtz_R(i,1,trajno);
     cvttemp_traj(i,j,kk)=cvt_traj(i,j,kk)/cvt_R_traj(i,1,kk);  % cross-correlation 
     % cvttempx_traj(i,j,trajno)=cvtx(i,j,trajno)/cvtx_R(i,1,trajno);
     % cvttempy_traj(i,j,trajno)=cvty(i,j,trajno)/cvty_R(i,1,trajno);
     % cvttempz_traj(i,j,trajno)=cvtz(i,j,trajno)/cvtz_R(i,1,trajno);
       end
    end
end

cvtfinalx_GR=zeros(deltarange+1,deltarange);
cvtfinal_GR=zeros(deltarange+1,deltarange);
cvtfinaly_GR=zeros(deltarange+1,deltarange);
cvtfinalz_GR=zeros(deltarange+1,deltarange);
cvtfinalx_R=zeros(deltarange+1,deltarange);
cvtfinal_R=zeros(deltarange+1,deltarange);
cvtfinaly_R=zeros(deltarange+1,deltarange);
cvtfinalz_R=zeros(deltarange+1,deltarange);

cvtfinal_GR_traj=zeros(deltarange+1,deltarange,trajno);
cvtfinal_R_traj=zeros(deltarange+1,deltarange,trajno);

for i=1:deltarange
    for j=1:deltarange
      cvtfinal_R(i+1,j)=cvttemp_R(j,i);
      cvtfinalx_R(i+1,j)=cvttempx_R(j,i);
      cvtfinaly_R(i+1,j)=cvttempy_R(j,i);
      cvtfinalz_R(i+1,j)=cvttempz_R(j,i);

      cvtfinal_R(1,j)=1;
      cvtfinalx_R(1,j)=1;
      cvtfinaly_R(1,j)=1;
      cvtfinalz_R(1,j)=1;
      cvtfinal_GR(i,j)=cvttemp(j,i);
      cvtfinalx_GR(i,j)=cvttempx(j,i);
      cvtfinaly_GR(i,j)=cvttempy(j,i);
      cvtfinalz_GR(i,j)=cvttempz(j,i);    
    end
end

for i=1:deltarange
    for j=1:deltarange
        for kk=1:trajno
      cvtfinal_R_traj(i+1,j,kk)=cvttemp_R_traj(j,i,kk);
     

      cvtfinal_R_traj(1,j,kk)=1;
     
      cvtfinal_GR_traj(i,j,kk)=cvttemp_traj(j,i,kk);
     
        end
    end
end


% %%
assignin('base','cvtfinal_R',cvtfinal_R)


assignin('base','cvtfinal_GR',cvtfinal_GR)
assignin('base','cvtfinal_R_traj',cvtfinal_R_traj)
assignin('base','cvtfinal_GR_traj',cvtfinal_GR_traj)


%len=size(Dfin_G,1);
%len=CutoffLen;
len=deltarange*2;
FrameT=0.05;
%labelt=(0:1:10)*FrameT;
labelt=(0:5:50)*FrameT;
matrix=nan(len/2+1,len/2+1);
for i= 1: len/2+1
        matrix(i,:)= (0:len/2)/i;
end
matrix=matrix';

trajsize=50;
figure
plot((1:len/2+1)*FrameT, cvtfinal_R(:,1:trajsize),LineWidth=2); %normally, no need to loop
%scatter((1:len/2+1)*FrameT, cvtfinal_R(:,1:trajsize),LineWidth=2); 
 colororder (turbo(size(cvtfinal_R,2)/2));
 colormap (turbo(size(cvtfinal_R,2)/2)); 
  colororder (turbo(trajsize));
 colormap (turbo(trajsize)); 
 colorbar('TickLabels',labelt);
 xlim([0 2.5])
% 
% figure
% plot(matrix(:,1:150), cvtfinal_R(:,1:150),LineWidth=2); %normally, no need to loop
% %scatter(matrix(:,1:250), cvtfinal_R(:,1:250),LineWidth=2); %normally, no need to loop
% 
%  colororder (turbo(size(cvtfinal_R,2)/2));
%  colormap (turbo(size(cvtfinal_R,2)/2)); 
%   colororder (turbo(50));
%  colormap (turbo(50)); 
%  colorbar('TickLabels',labelt);
% xlim([0 5])

% figure
% trajsize=10;
% labelt=(0:1:10)*FrameT;
% plot((1:len/2+1)*FrameT, cvtfinal_GR(:,1:10),LineWidth=2); %normally, no need to loop
%  % colororder (turbo(size(cvtfinal_GR,2)/2));
%  % colormap (turbo(size(cvtfinal_GR,2)/2)); 
%   colororder (turbo(trajsize));
%  colormap (turbo(trajsize)); 
%  colorbar('TickLabels',labelt,FontSize=15);
%xlim([0 2.5])

figure
trajsize=50;
labelt=(1:1:10)*FrameT*10;
plot(matrix(:,1:trajsize), cvtfinal_GR(:,1:trajsize),LineWidth=2); %normally, no need to loop
 % colororder (turbo(size(cvtfinal_GR,2)/2));
 % colormap (turbo(size(cvtfinal_GR,2)/2)); 

  colororder (turbo(trajsize));
 colormap (turbo(trajsize)); 
% colorbar('TickLabels',labelt,FontSize=15);
 colorbar('Ticks', ((1/trajsize:1/trajsize:1))*5,'TickLabels',labelt,FontSize=15);
xlim([0 2.5])

figure
trajsize=5;
curve=[1 5 10 50 100];
labelt=curve*FrameT;
plot(matrix(:,curve), cvtfinal_GR(:,curve),LineWidth=4); %normally, no need to loop
 % colororder (turbo(size(cvtfinal_GR,2)/2));
 % colormap (turbo(size(cvtfinal_GR,2)/2)); 
 %  colororder (turbo(trajsize));
 % colormap (turbo(trajsize)); 
 colororder (lines(trajsize));
 colormap (lines(trajsize)); 
%  colorbar('TickLabels',labelt,FontSize=15);
% colorbar('Ticks', [1,2,3,4,5]*0.2-0.1,'TickLabels',labelt,FontSize=15);
 xlim([0 2.5])
 ylim([-.5 1])
 mse=[];
 mse(:,1)=(0:1:2)/0.01;
% colorbar('TickLabels',labelt,FontSize=15);
  fontsize(60,"points");
for j=1:trajsize
    legend_text{j}=strcat('\delta=',num2str(curve(j)*0.05), 's');
end
set(gca,'FontSize',30)  
set(gca,'linewidth',4)  
set(gca,'TickLength', [0.02 0.02])
legend(legend_text)
xlabel('\tau/\delta','FontSize',30)
ylabel('C_{vv}^{(\delta)}(\tau,\Deltan)/C_{vv}^{(\delta)}(0,0)','FontSize',30)

figure
trajsize=5;
curve=[1 5 10 50 100];
labelt=curve*FrameT;
for kk=1:trajno
   plot(matrix(:,curve), cvtfinal_GR_traj(:,curve,kk),LineWidth=4);
   hold on;
end
xlim([0 2.5])
 ylim([-.5 1])

 % figure
 % plot((1:1:size(cvtfinal_GR,2))*0.05, cvtfinal_GR(1,:),LineWidth=2);
 % xlim([0 5])
 %  xlabel('delta(s)');
 % ylabel('CVV(t=0)');
 % fontsize(35,"points");

 %   figure
 % plot(log((1:1:size(cvtfinal_GR,2))*0.05), log(cvtfinal_GR(1,:)),LineWidth=2);
 % xlim(log([0 5]))
 %  xlabel('log(delta(s))');
 % ylabel('log(CVV(t=0))');
 % fontsize(35,"points");
 % x=log((1:1:size(cvtfinal_GR,2))*0.05);
 % y=log(cvtfinal_GR(1,:));



x=(0:1:CutoffLen-1)*FrameT;
assignin('base','time',x)
assignin('base','matrix',matrix);

