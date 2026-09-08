% Thomas Lampo, Andrew Kennard, and Andrew Spakowitz
% November 2015
%
% This script plots a set of Velocity-Velocity Correlation Function (VVCF)
% curves for a Rouse polymer in a viscoelastic medium using numerical 
% tables and interpolation.  The numerical tables are calculated 
% as described in the manuscript by the above authors as printed in
% Biophysical Journal, specifically using the VVCF defined by Eq. 10 in the
% intermediate time regime described by Eq. 12.
%
% Usage:
% alpha should be a number between 0.25 and 1.  alpha is the power law
% exponent associated with the mean square displacement of a particle in a
% viscoelastic medium.
%
% delta_range is the set of delta/t_(delta n) values that you wish to 
% examine.  delta_range should be a vector whose values lie between 10^-3 
% and 10^3.
% 
% plot_range is the x-axis range of the plot for t/delta.  This should be 
% a 2 element vector whose minimum is greater than or
% equal to 0 and whose maximum is less than or equal to 5.  If no value is
% defined when calling calc_vel_corr, the default value is [0 5].
%
% MAIN ADJUSTABLE PARAMETERS:
alpha=0.9; 
% t_delta_n=0.25;% [0.25:0.25:2.5]
F = [];
    ii = 1;
for t_delta_n = 0.01:0.01:1
delta_range=[0.05 0.25 0.5 2.5 5]/t_delta_n; %logspace(-2,1,4); 
%t_delta_n=96;% [0.25:0.25:2.5]
%delta_range=[1 2 5 10]*24/t_delta_n; %logspace(-2,1,4); 
plot_range=[0 2.5]; 
%
%
% round down/up the low/high values in plot_range to get the initial and 
% final time points to be used in the plot.  The tables are calculated in 
% time increments of t/delta of 0.01.
t1=floor(plot_range(1)*100)+1;
t2=ceil(plot_range(2)*100)+1;
%
% set the color range for the plot curves
%colors=copper(length(delta_range));
%colors=turbo(length(delta_range));
colors=lines(length(delta_range));
%
% use the calc_vel_corr script to calculate the array of VVCF curves
vvcf=calc_vel_corr(alpha,delta_range,plot_range);
%
%
f = figure(1); f.Position=[100 100 1000 900];
%f = figure('Position',[100 100 1000 900]);
% 
% the time vector for t/delta (again, the tables are calculated in t/delta
% increments of 0.01).
time=((t1:t2)-1)./100;
%
%
% the loop for plotting the array of VVCF curves 
for j=1:length(vvcf)
    plot(time,vvcf{j},'color',colors(j,:),'LineWidth',3)
    hold on
    legend_text{j}=strcat('\delta/\tau_{\Deltan}=',num2str(delta_range(j)));
end
%
% Set figure properties and axis labels
axis([0 plot_range(2) -0.5 1])
xlabel('\tau/\delta','FontSize',20)
ylabel('C_{vv}^{(\delta)}(\tau,\Deltan)/C_{vv}^{(\delta)}(0,0)','FontSize',20)
set(gca,'FontSize',35)  
set(gca,'TickLength', [0.03 0.03])
set(gca,'linewidth',4)  
set(gca,'TickDir','out');
set(gca().Title, 'String', ['\tau_{\Delta}_n = ' num2str(t_delta_n)]);
legend(legend_text)
hold off
F(:,:,:,ii) = getframe(f).cdata;
% pause(0.2);
ii = ii+1;
end

%%
[FileName,PathName] = uiputfile({'*.mp4';'*.*'},'Save movie');
%%
v = VideoWriter([PathName FileName],'MPEG-4'); %file name to record, h264 encoding
v.Quality = 100; %video quality (from 0 to 100)
v.FrameRate = 10; %frame rate; more options: https://www.mathworks.com/help/matlab/ref/videowriter.html
open(v);
for i = 1:size(F,4)
       writeVideo(v,uint8(F(:,:,:,i)));
end
close(v);