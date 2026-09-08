function []= v_autocorrelation()
% author: Yanyu Zhu created in 12/8/2016
% the code calculate the velocity auto-correlation  between the with the
% input of Dfin and Dfiny

prompt = {'time betwee frames:',...
               'length of pixel'}; 
        u_name = 'Input parameters';
        numlines = 1;
        defaultanswer = {'0.02','.001'};
        options.Resize = 'on';
        options.WindowStyle = 'normal';
        options.Interpreter = 'tex';
        user_var = inputdlg(prompt,u_name,numlines,defaultanswer,options);
        FrameT= str2double(user_var{1});
        pixl= str2double(user_var{2});

     % reading the Dfin and Dfiny Matrices from the workspace   
        Dfin=evalin('base','Dfin');
        Dfiny=evalin('base','Dfiny');

trajno=size(Dfin,2);%010
CutoffLen=size(Dfin,1);

delta=FrameT;
cvt=zeros(CutoffLen/2,CutoffLen-1);
cvtx=zeros(CutoffLen/2,CutoffLen-1);
cvty=zeros(CutoffLen/2,CutoffLen-1);

 for I=1 % possible ways to have lag time
   for d=1    %  calculate velocity delta
     vr = zeros(CutoffLen-d,trajno);
     vy = zeros(CutoffLen-d,trajno);
     vx = zeros(CutoffLen-d,trajno);
     cv=zeros(CutoffLen-d-I,trajno);
      cv_nor=zeros(CutoffLen-d-I,trajno);
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
            cv(j-I,i)=vx(j,i)*vx(j-I,i)+vy(j,i)*vy(j-I,i); % I is like tao,lag time
            cv_nor(j-I,i)=cv(j-I,i)/(sqrt(vx(j,i)*vx(j,i)+vy(j,i)*vy(j,i))*sqrt(vx(j-I,i)*vx(j-I,i)+vy(j-I,i)*vy(j-I,i)));
            cvx(j-I,i)=vx(j,i)*vx(j-I,i);
            cvy(j-I,i)=vy(j,i)*vy(j-I,i);
            
        end
    end
    cvt(d,I+1)=nanmean(nanmean(cv));
    cvtx(d,I+1)=nanmean(nanmean(cvx));
    cvty(d,I+1)=nanmean(nanmean(cvy));
   end
end
% to get the first point as zero assigning a new matrix
avg=5;

 cv_avg=zeros(CutoffLen-2-avg,trajno);
 cv_avg_nor=zeros(CutoffLen-2-avg,trajno);


for k=1:CutoffLen-2-avg
    for i=1:trajno
        cv_avg(k,i)=nanmean(cv(k:k+avg-1,i));
        cv_avg_nor(k,i)=nanmean(cv_nor(k:k+avg-1,i));
    end
end

assignin('base','cv',cv)
assignin('base','cv_avg',cv_avg)
assignin('base','cv_avg_nor',cv_avg_nor)
assignin('base','cv_nor',cv_nor)


x=(0:1:CutoffLen-1)*FrameT;
assignin('base','time',x)

for i=1:1
figure
plot(1:1:298,cv(:,i));
hold on;
plot(1:1:293,cv_avg(:,i));
end

for i=1:1
    figure
    plot(1:1:298,cv_nor(:,i));
    hold on; 
plot(1:1:293,cv_avg_nor(:,i));
end


end