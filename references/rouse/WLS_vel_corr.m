% experimental curves
%curve = [1 5 10 50 100];
curve=[1 2 5 10];
%curve = [1 2 5 10 50 100];
%curve = 1:1:100;
Xexp = matrix(:,curve);
Yexp = cvtfinal_GR(:,curve);

% theoretical curves
alpha = .91;
plot_range = [0 2.5];
t1 = floor(plot_range(1) * 100) + 1;
t2 = ceil(plot_range(2) * 100) + 1;
Xtheor = (((t1:t2) - 1) ./ 100)';

% weighted mean squared error
%t_delta_n = 0.01:0.01:5; % t_delta_n set up here
t_delta_n = 1:1:1000; % t_delta_n set up here for 200kb
WMSE = NaN(size(t_delta_n));

for jj = 1:length(t_delta_n)

    %delta_range = [0.05 0.25 0.5 2.5 5]/t_delta_n(jj);
    delta_range = curve*24/t_delta_n(jj);
    Ytheor = transpose(cell2mat(transpose(calc_vel_corr_bug(alpha,delta_range,plot_range))));
    [Lia,Locb] = ismember(Xexp, Xtheor);
    Wsum = NaN(size(t_delta_n(jj)));
    for ii = 1 : size(Ytheor,2)
        Wsum(ii) = sum((Yexp(Lia(:,ii),ii)-Ytheor(Locb(Lia(:,ii),ii),ii)).^2)/sum(Lia(:,ii)); % weighted sum
    end
    WMSE(jj) = sum(Wsum);

end
[~, MinIndex] = min(WMSE); % MinIndex is the index of the lowest WMSE
Best_t_delta_n = t_delta_n(MinIndex);