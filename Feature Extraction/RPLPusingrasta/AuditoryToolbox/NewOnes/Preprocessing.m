function [y, newfs]= Preprocessing(x, fs, n, k) 
    % removes samples less than .009 envelope and samples the audiofile at 16khz
    % % % % % % starting here  

    [r,c]= size(x);
    if c>1

        sum=zeros([r,1]);
        for j=1:c          
            sum=sum+x(:,j);
        end
        x=sum/c;        
    end
    x=double(x);
    % Find the envelope by taking a moving max operation, imdilate.
    envelope = imdilate(abs(x), true(1501, 1));
    % Find the quiet parts.
    quietParts = envelope < 0.001; % Or whatever value you want.
    % Cut out quiet parts 
    xEdited = x; % Initialize
    xEdited(quietParts) = [];
    % Find the ratio of sampling at 16000 Hz
    [P,Q] = rat(16e3/fs);
    % Sample at 16000 Hz
    xnew = resample(xEdited,P,Q);
    % write audio file
%     outputfilename=strcat('C:\Users\Muna\Desktop\audio\',int2str(n),int2str(k),'.wav');
%     audiowrite(outputfilename, xnew, 16000);
%     y=xnew(1:round(length(xnew)/2));
    y=xnew;
    newfs=16000;

end
