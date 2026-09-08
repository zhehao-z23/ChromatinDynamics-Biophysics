function []= v_autocorrelation()
% author: Yanyu Zhu created in 12/8/2016
% the code calculate the velocity auto-correlation  between the with the
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
        Dfin=evalin('base','Dfin_P');
        Dfiny=evalin('base','Dfiny_P');

trajno=size(Dfin,2);
CutoffLen=size(Dfin,1);

delta=FrameT;
cvt=zeros(CutoffLen/2,CutoffLen-1);
cvtx=zeros(CutoffLen/2,CutoffLen-1);
cvty=zeros(CutoffLen/2,CutoffLen-1);
 for I=0:CutoffLen/2 % possible ways to have lag time
   for d=1:CutoffLen/2    %  calculate velocity delta
     vr = zeros(CutoffLen-d,trajno);
     vy = zeros(CutoffLen-d,trajno);
     vx = zeros(CutoffLen-d,trajno);
     cv=zeros(CutoffLen-d-I,trajno);
     cvx=zeros(CutoffLen-d-I,trajno);
     cvy=zeros(CutoffLen-d-I,trajno);
    for i=1:trajno
        for j=d+1:CutoffLen
            if Dfiny(j,i)~=0
                vx(j-d,i)=(Dfin(j,i)-Dfin((j-d),i))/(delta*d);
                vy(j-d,i)=(Dfiny(j,i)-Dfiny((j-d),i))/(delta*d);
             %   vr(j-I,i)=vx(j-I,i)+vy(j-I,i);
            end
        end
    end  
    for j=1+I:CutoffLen-d
        for i=1:trajno
            cv(j-I,i)=vx(j,i)*vx(j-I,i)+vy(j,i)*vy(j-I,i);  % I is like tao,lag time
            cvx(j-I,i)=vx(j,i)*vx(j-I,i);
            cvy(j-I,i)=vy(j,i)*vy(j-I,i);
        end
    end
    cvt(d,I+1)=nanmean(nanmean(cv));
    cvtx(d,I+1)=nanmean(nanmean(cvx));
    cvty(d,I+1)=nanmean(nanmean(cvy));

    % cvt(d,I+1) = mean(cv(:), 'omitnan');
    % cvtx(d,I+1) = mean(cvx(:), 'omitnan');
    % cvty(d,I+1) = mean(cvy(:), 'omitnan');
   end
end
% to get the first point as zero assigning a new matrix

cvttemp=zeros(CutoffLen/2,CutoffLen-1);
cvttempx=zeros(CutoffLen/2,CutoffLen-1);
cvttempy=zeros(CutoffLen/2,CutoffLen-1);

for j=1:CutoffLen/2
    for i=1:CutoffLen/2

     cvttemp(i,j)=cvt(i,j+1)/cvt(i,1);
     cvttempx(i,j)=cvtx(i,j+1)/cvtx(i,1);
     cvttempy(i,j)=cvty(i,j+1)/cvty(i,1);
    end
end
cvtfinalx=zeros(CutoffLen/2+1,CutoffLen/2);
cvtfinal=zeros(CutoffLen/2+1,CutoffLen/2);
cvtfinaly=zeros(CutoffLen/2+1,CutoffLen/2);
for i=1:CutoffLen/2
    for j=1:CutoffLen/2
      cvtfinal(i+1,j)=cvttemp(j,i);
      cvtfinalx(i+1,j)=cvttempx(j,i);
      cvtfinaly(i+1,j)=cvttempy(j,i);
      cvtfinal(1,j)=1;
      cvtfinalx(1,j)=1;
      cvtfinaly(1,j)=1;
      
    end
end
%%
len=size(Dfin,1);
FrameT=1;
labelt=(0:1:20)*FrameT;
delta_size=20;
cmap = turbo(delta_size);

%%
figure
plot((1:len/2+1)*FrameT, cvtfinal(:,1:50),LineWidth=2); %normally, no need to loop
 colororder (turbo(size(cvtfinal,2)/2));
 colormap (turbo(size(cvtfinal,2)/2)); 
 
 %  colororder (turbo(10));
 % colormap (turbo(10)); 
 colorbar('TickLabels',labelt);
% xlim([0 10])
% ylim([-0.5 1])
set(gca,'FontSize',45)  
set(gca,'linewidth',4) 
xlabel('tao (s)');
ylabel('cv(tao)/cv(0)');
fontsize(30,"points");
%%

matrix=nan(len/2+1,len/2+1);
for i= 1: len/2+1
        matrix(i,:)= (0:len/2)/i;
end
matrix=matrix';
% figure
% plot(matrix(:,1:delta_size), cvtfinal(:,1:delta_size),LineWidth=2); %normally, no need to loop
% %scatter(matrix(:,1:50), cvtfinal(:,1:50)); 
%  colororder (turbo(size(cvtfinal,2)/2));
%  colormap (turbo(size(cvtfinal,2)/2)); 
%   colororder (turbo(delta_size));
%  colormap (turbo(delta_size)); 
%  colorbar('TickLabels',labelt);


cmap = turbo(delta_size);

figure; hold on
for d = 1:delta_size
    plot(matrix(:,d), cvtfinal(:,d), ...
        'LineWidth', 2, ...
        'Color', cmap(d,:));
end

xlim([0 2.5])
ylim([-1 1])
xlabel('\tau/\delta', 'FontSize', 50)
ylabel('C_v^{(\delta)}(\tau)/C_v^{(\delta)}(0)', 'FontSize', 50)
set(gca,'FontSize',30,'LineWidth',4)

% Color scale visually runs from 0 to delta_size
colormap(cmap)
clim([0 delta_size])

% Automatically make ~5 intervals / 6 labels, including zero
nIntervals = 4;
step = max(1, ceil(delta_size / nIntervals));
tickTime = 0:step:delta_size;

if tickTime(end) ~= delta_size
    tickTime = [tickTime, delta_size];
end

cb = colorbar;
cb.Ticks = tickTime;
cb.TickLabels = compose('%g s', tickTime);
title(cb, '\delta', 'FontSize', 30, 'Interpreter', 'tex');


%% Scatter VAC collapse plot with an automatic delta colorbar

% %plot(matrix(:,1:50), cvtfinal(:,1:50),LineWidth=2); %normally, no need to loop
% scatter(matrix(:,1:delta_size), cvtfinal(:,1:delta_size),'filled'); 
%  colororder (turbo(size(cvtfinal,2)/2));
%  colormap (turbo(size(cvtfinal,2)/2)); 
%   colororder (turbo(delta_size));
%  colormap (turbo(delta_size)); 
%  colorbar('TickLabels',labelt);

%  hold on;


% %legend('fit with alpha=0.43')


figure; hold on

% Draw each delta separately, so its scatter color exactly matches the colorbar
for d = 1:delta_size
    scatter(matrix(:,d), cvtfinal(:,d), 28, ...
        'filled', ...
        'MarkerFaceColor', cmap(d,:), ...
        'MarkerEdgeColor', 'none');
end

% fBM theoretical prediction


 step=0.01;
x1=0:step:2.5;
beta=0.38;
y_fit=0.5*((abs(1-x1)).^(beta)+(1+x1).^beta-2*(x1.^beta));
plot(x1,y_fit,'b',LineWidth=3);
xlim([0 2.5])
ylim([-0.5 1])


xlabel('\tau/\delta', 'FontSize', 50)
ylabel('C_{v}^{(\delta)}(\tau)/C_{v}^{(\delta)}(0)', 'FontSize', 50)
set(gca, 'FontSize', 30, 'LineWidth', 4)

% Colorbar: visual range 0–delta_size; actual curves are delta = 1:delta_size
colormap(cmap)
clim([0 delta_size])

nIntervals = 4;  % Gives approximately five intervals / six labels
tickStep = max(1, ceil(delta_size / nIntervals));
tickTime = 0:tickStep:delta_size;

if tickTime(end) ~= delta_size
    tickTime = [tickTime, delta_size];
end

cb = colorbar;
cb.Ticks = tickTime;
cb.TickLabels = compose('%g s', tickTime);
title(cb, '\delta', 'FontSize', 30, 'Interpreter', 'tex');
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