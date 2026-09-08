function []= v_autocorrelation()
% author: Yanyu Zhu created in 12/8/2016
% the code calculate the velocity auto-correlation  between the with the
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
% start=1; 
% d=1550;
%  col=2;
  Dfin=evalin('base','Dfin_rep'); % rep FOS-P Cy5; non-rep FOS-E Cy3
  Dfiny=evalin('base','Dfiny_rep');
  Dfinz=evalin('base','Dfinz_rep');
% 
% figure
% plot(1:4500, Dfin_rep(1:4500,1));
% hold on;
% plot(1:4500, Dfiny_rep(1:4500,1));
% plot(1:4500, Dfinz_rep(1:4500,1));

  %   Dfin=evalin('base','Dfin_rep'); % rep FOS-P Cy5; non-rep FOS-E Cy3
  % Dfiny=evalin('base','Dfiny_rep');
  % Dfinz=evalin('base','Dfinz_rep');

  % Dfin=evalin('base','Dfin_nonrep'); % rep FOS-P Cy5; non-rep FOS-E Cy3
  % Dfiny=evalin('base','Dfiny_nonrep');
  % Dfinz=evalin('base','Dfinz_nonrep');

trajno=size(Dfin,2);
CutoffLen=size(Dfin,1);
deltarange=100;
delta=FrameT;
cvt=zeros(deltarange,CutoffLen-1);
cvtx=zeros(deltarange,CutoffLen-1);
cvty=zeros(deltarange,CutoffLen-1);
cvtz=zeros(deltarange,CutoffLen-1);

 for I=0:deltarange % possible ways to have lag time
   for d=1:deltarange   %  calculate velocity delta
     vr = zeros(CutoffLen-d,trajno);
     vy = zeros(CutoffLen-d,trajno);
     vx = zeros(CutoffLen-d,trajno);
     vz = zeros(CutoffLen-d,trajno);
     cv=zeros(CutoffLen-d-I,trajno);
     cvx=zeros(CutoffLen-d-I,trajno);
     cvy=zeros(CutoffLen-d-I,trajno);
     cvz=zeros(CutoffLen-d-I,trajno);
    for i=1:trajno
        for j=d+1:CutoffLen
            if Dfiny(j,i)~=0
                vx(j-d,i)=(Dfin(j,i)-Dfin((j-d),i))/(delta*d);
                vy(j-d,i)=(Dfiny(j,i)-Dfiny((j-d),i))/(delta*d);
                vz(j-d,i)=(Dfinz(j,i)-Dfinz((j-d),i))/(delta*d);
             %   vr(j-I,i)=vx(j-I,i)+vy(j-I,i);
            end
        end
    end  
    for j=1+I:CutoffLen-d
        for i=1:trajno
            cv(j-I,i)=vx(j,i)*vx(j-I,i)+vy(j,i)*vy(j-I,i)+vz(j,i)*vz(j-I,i);  % I is like tao,lag time
            cvx(j-I,i)=vx(j,i)*vx(j-I,i);
            cvy(j-I,i)=vy(j,i)*vy(j-I,i);
            cvz(j-I,i)=vz(j,i)*vz(j-I,i);
        end
    end
    cvt(d,I+1)=nanmean(nanmean(cv));
    cvtx(d,I+1)=nanmean(nanmean(cvx));
    cvty(d,I+1)=nanmean(nanmean(cvy));
    cvtz(d,I+1)=nanmean(nanmean(cvz));

    % cvt(d,I+1)=mean(cv,"all","omitnan");
    % cvtx(d,I+1)=mean(cvx,"all","omitnan");
    % cvty(d,I+1)=mean(cvy,"all","omitnan");
    % cvtz(d,I+1)=mean(cvz,"all","omitnan");
   end
 end

 
% to get the first point as zero assigning a new matrix

cvttemp=zeros(deltarange,CutoffLen-1);
cvttempx=zeros(deltarange,CutoffLen-1);
cvttempy=zeros(deltarange,CutoffLen-1);
cvttempz=zeros(deltarange,CutoffLen-1);

for j=1:deltarange
    for i=1:deltarange

     cvttemp(i,j)=cvt(i,j+1)/cvt(i,1);
     cvttempx(i,j)=cvtx(i,j+1)/cvtx(i,1);
     cvttempy(i,j)=cvty(i,j+1)/cvty(i,1);
     cvttempz(i,j)=cvtz(i,j+1)/cvtz(i,1);
    end
end
cvtfinalx=zeros(deltarange+1,deltarange);
cvtfinal=zeros(deltarange+1,deltarange);
cvtfinaly=zeros(deltarange+1,deltarange);
cvtfinalz=zeros(deltarange+1,deltarange);
for i=1:deltarange
    for j=1:deltarange
      cvtfinal(i+1,j)=cvttemp(j,i);
      cvtfinalx(i+1,j)=cvttempx(j,i);
      cvtfinaly(i+1,j)=cvttempy(j,i);
      cvtfinalz(i+1,j)=cvttempz(j,i);
      cvtfinal(1,j)=1;
      cvtfinalx(1,j)=1;
      cvtfinaly(1,j)=1;
      cvtfinalz(1,j)=1;
    end
end
%%
deltarange=100;
len=deltarange*2;
%len=1000;
FrameT=0.05;
labelt=(0:5:50)*FrameT;
figure
plot((1:len/2+1)*FrameT, cvtfinal(:,1:50),LineWidth=2); %normally, no need to loop
 colororder (turbo(size(cvtfinal,2)/2));
 colormap (turbo(size(cvtfinal,2)/2)); 
  colororder (turbo(50));
 colormap (turbo(50)); 
 colorbar('TickLabels',labelt);
xlim([0 2.5])
xlabel('tao (s)');
ylabel('cv(tao)/cv(0) ');
fontsize(30,"points");
%%
matrix=nan(len/2+1,len/2+1);
for i= 1: len/2+1
        matrix(i,:)= (0:len/2)/i;
end
matrix=matrix';
figure
plot(matrix(:,1:50), cvtfinal(:,1:50),LineWidth=2); %normally, no need to loop
%scatter(matrix(:,1:50), cvtfinal(:,1:50)); 
 colororder (turbo(size(cvtfinal,2)/2));
 colormap (turbo(size(cvtfinal,2)/2)); 
  colororder (turbo(50));
 colormap (turbo(50)); 
 colorbar('TickLabels',labelt);
xlim([0 2.5])
ylim([-1 1])
%%
figure
%plot(matrix(:,1:50), cvtfinal(:,1:50),LineWidth=2); %normally, no need to loop
scatter(matrix(:,1:50), cvtfinal(:,1:50)); 
 colororder (turbo(size(cvtfinal,2)/2));
 colormap (turbo(size(cvtfinal,2)/2)); 
  colororder (turbo(50));
 colormap (turbo(50)); 
 colorbar('TickLabels',labelt);
 hold on;
 step=0.01;
x1=0:step:2.5;
beta=0.3;
y_fit=0.5*((abs(1-x1)).^(beta)+(1+x1).^beta-2*(x1.^beta));
plot(x1,y_fit,'b',LineWidth=3);
xlim([0 2.5])
ylim([-0.4 1])
set(gca,'FontSize',45)  
set(gca,'linewidth',4)  
%legend('fit with alpha=0.43')
xlabel('\tau/\delta','FontSize',50)
ylabel('C_{v}^{(\delta)}(\tau)/C_{v}^{(\delta)}(0)','FontSize',50)


%%
% figure(4)
% for kk=1:50
%    colormap(hsv)
%     plot(1:101, cvtfinal(:,kk));
%     colororder (turbo(size(cvtfinal,2)));
%     colorbar;
%    hold on;
%    xlim([0 75])
% end 

assignin('base','cvt',cvt)
assignin('base','cvttemp',cvttemp)
assignin('base','cvtfinal',cvtfinal)
assignin('base','cvtfinalx',cvtfinalx)
assignin('base','cvtfinaly',cvtfinaly)

x=(0:1:CutoffLen-1)*FrameT;
assignin('base','time',x)



end