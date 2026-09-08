% experimental curves
%curve = 1:1:100; % for RGP
%curve = 1:1:10; % for RGP
curve=[1 5 10 50 100];
%curve=[1 2 5 10];  % for RGP
%FrameT=24;
FrameT=0.05;
Xexp = matrix(:,curve);
% theoretical curves
alpha = 1;
plot_range = [0 2.5];
t1 = floor(plot_range(1) * 100) + 1;
t2 = ceil(plot_range(2) * 100) + 1;
Xtheor = (((t1:t2) - 1) ./ 100)';

% weighted mean squared error
t_delta_n = 0.1:0.1:50; % t_delta_n set up here for 30 kb
%t_delta_n = 1:1:5000; % t_delta_n set up here for 200kb
for kk=1:trajno
Yexp = cvtfinal_GR_traj(:,curve,kk);
%Yexp = cvtfinal_GR_2mean(:,curve);

WMSE = NaN(size(t_delta_n));
[Lia,Locb] = ismember(round(Xexp, 2), Xtheor);
parfor jj = 1:length(t_delta_n) % change to parfor for parallel processing
    delta_range = curve*FrameT/t_delta_n(jj);
    Ytheor = transpose(cell2mat(transpose(calc_vel_corr_new(alpha,delta_range,plot_range))));
    Wsum = NaN(size(Ytheor,2),1);
    for ii = 1 : size(Ytheor,2)
        Wsum(ii) = sum((Yexp(Lia(:,ii),ii)-Ytheor(Locb(Lia(:,ii),ii),ii)).^2)/sum(Lia(:,ii)); % weighted sum
    end
    WMSE(jj) = sum(Wsum) / length(curve);
end
[~, MinIndex] = min(WMSE); % MinIndex is the index of the lowest WMSE
Best_t_delta_n(kk) = t_delta_n(MinIndex);

end


% figure
% plot(t_delta_n,WMSE,LineWidth=3);
% legend("canonical group site 1&2 284 kb")
% %ylim([0 0.1])
% set(gca,'linewidth',3)  
% set(gca,'FontSize',35) 
% xlabel('\tau_{\Deltan}','FontSize',35)
% ylabel('WMSE','FontSize',35)
% hold on;