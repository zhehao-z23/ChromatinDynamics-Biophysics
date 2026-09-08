function VVCF=calc_vel_corr(alpha,delta_range,plot_range)
%
% Thomas Lampo, Andrew Kennard, and Andrew Spakowitz
% November 2015
% 
% This script calculates a set of velocity-velocity correlation function
% (VVCF) curves for a Rouse polymer in a viscoelastic medium using 
% numerical tables and interpolation.  The numerical tables are calculated 
% as described in the manuscript by the above authors as printed in
% Biophysical Journal, specifically using the VVCF defined by Eq. 10 in the
% intermediate time regime described by Eq. 12.
%
% The script output is an array of vectors for a set of VVCF curves
%
if nargin==2
    plot_range=[0 5];
end
%
% check for common input errors and output an error message if an error is
% detected
%
if nargin<2
    sprintf('Not enough inputs.  Input (alpha,delta_range,plot_range)')
elseif alpha>1 || alpha<0.25
    sprintf('alpha must be between 0.25 and 1')
elseif min(delta_range)<10^-3 || max(delta_range)>10^3
    sprintf('delta_range must contain elements between 10^-3 and 10^3')
elseif min(plot_range)<0 || max(plot_range)>5
    sprintf('plot_range should be a 2-element vector with elements between 0 and 5')
% 
% if common errors are not detected, proceed with the interpolation
% calculation to produce the velocity-velocity correlation curves
%
else
    %
    % the interpolation script is split into the alpha==1 and alpha=/=1
    % cases 
    %
    if alpha==1
        % for alpha==1 case
        %
        for j=1:length(delta_range)
            % load in the table for alpha=1
            table_file=load('tables/table1000.mat');
            vac_table=table_file.VAC_Table;
            % find the initial and final time points in the table based on
            % plot_range.  The tables are calculated in time increments of
            % t/delta of 0.01.
            t1=floor(plot_range(1)*100)+1;
            t2=ceil(plot_range(2)*100)+1;
            % use delta to find the relevant vectors in the table to
            % perform the linear interpolation
            d_true=(log(delta_range(j))/log(10)+3)*4+1;
            d1=floor((log(delta_range(j)*10^3)/log(10))*4+1);
            d2=ceil((log(delta_range(j)*10^3)/log(10))*4+1);
            % calculate the VVCF using linear interpolation
            if d1==d2
                vvcf=vac_table(d1,t1:t2);
                VVCF{j}=vvcf;
            else
                vvcf_1=vac_table(d1,t1:t2);
                vvcf_2=vac_table(d2,t1:t2);
                prop=(d_true-d1)/(d2-d1);
                vvcf=prop*vvcf_2+(1-prop)*vvcf_1;
                VVCF{j}=vvcf;
            end
        end
    else
        % for alpha=/=1
        %
        % use alpha to find the relevant tables with which to perform the
        % linear interpoloation.  Each table is calculated for alpha in 
        % increments of 0.025
        alpha_1=(floor(alpha/.025))*.025;
        alpha_2=alpha_1+.025;
        % 
        % 
        for j=1:length(delta_range)
            % load the relevant tables
            table_str1=strcat('tables/table0',num2str(alpha_1*1000),'.mat');
            table_file1=load(table_str1);
            vac_table_1=table_file1.VAC_Table;
            table_str2=strcat('tables/table0',num2str(alpha_2*1000),'.mat');
            table_file2=load(table_str2);
            vac_table_2=table_file2.VAC_Table;
            % find the initial and final time points in the table based on
            % plot_range
            t1=floor(plot_range(1)*100)+1;
            t2=ceil(plot_range(2)*100)+1;
            % use delta to find the relevant vectors in the table to
            % perform the linear interpolation
            d_true=(log(delta_range(j))/log(10)+3)*4+1;
            d1=floor((log(delta_range(j)*10^3)/log(10))*4+1);
            d2=ceil((log(delta_range(j)*10^3)/log(10))*4+1);
            % calculate the VVCF using linear interpolation
            if d1==d2
                vvcf_1=vac_table_1(d1,t1:t2);
                vvcf_2=vac_table_2(d1,t1:t2);
            else
                vacf_1a=vac_table_1(d1,t1:t2);
                vacf_1b=vac_table_1(d2,t1:t2);
                vacf_2a=vac_table_2(d1,t1:t2);
                vacf_2b=vac_table_2(d2,t1:t2);
                prop=(d_true-d1)/(d2-d1);
                vvcf_1=prop.*vacf_1a+(1-prop).*vacf_1b;
                vvcf_2=prop.*vacf_2a+(1-prop).*vacf_2b;         
            end
            if alpha_1==alpha_2
                vvcf=vvcf_1;
                VVCF{j}=vvcf;
            else
                prop2=(alpha-alpha_1)/(alpha_2-alpha_1);
                vvcf=prop2.*vvcf_1+(1-prop2).*vvcf_2;
                VVCF{j}=vvcf;
            end
        end
    end
end    
end
    