addpath('AuditoryToolbox');
addpath('AuditoryToolbox\NewOnes');

% myFolder = 'G:\8th semester dropbox extras\ThesisDataset\Test_set';
myFolder = 'G:\8th semester dropbox extras\ThesisDataset\Train_set';
% ResultFolder= 'RPLPWaveletTrainResultFolder';
% ResultFolder= 'RPLPTrainResultFolder';
ResultFolder= 'PLPTrainResultFolder';


% ResultFolder= 'RPLPWaveletTestResultFolder';
% ResultFolder= 'RPLPTestResultFolder';
% ResultFolder= 'PLPTestResultFolder';

% Check to make sure that folder actually exists.  Warn user if it doesn't
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
if ~isdir(ResultFolder)
   mkdir (ResultFolder);
end
% resultfile='resultfile.txt';
% resultFileId= fopen(resultfile, 'a');

% y_file= 'y_file.txt';
% y_fileId= fopen(y_file, 'a');
maximum=0;
for n = 0:9
    % Get a list of all files in the folder with the desired file name pattern.
    subfolder=int2str(n);
    filePattern = fullfile(myFolder,subfolder,'*.wav'); % Change to whatever pattern you need.
    theFiles = dir(filePattern);
    subDirectoryPath=strcat(ResultFolder,'\', int2str(n));
    if ~isdir(subDirectoryPath)
        mkdir (subDirectoryPath);
    end
    
    for k = 1 : length(theFiles)
      baseFileName = theFiles(k).name;
      
      fullFileName = fullfile(myFolder,subfolder, baseFileName);
      % fprintf(1, 'Now reading %s\n', fullFileName);
      % Now do whatever you want with this file name,
      
      % x: signal
      % fs: sampling frequency
      [x,fs] = audioread(fullFileName);
      [x,fs]= Preprocessing(x, fs, n, k);
      
      % N: frame length 
      N= 400;
      % M: frame rate in samples here 10 ms 
      M= 160;
      % Nband: num. of (effective) critical bands
      Nband=257;
      % P: auditory power spectrum (Nband+2)x(Nframes) : 
      p=5; %hermansky (5th order) max= 15, if PLP model order is larger than the default of 8, append deltas and double-deltas onto the cepstral vectors
      Q=50; %Q cepstral coeficients for each frame
      % [c,a,P]=PLP(x,N,M,Nband,fs,p,Q);
      %  computs vector "c" with Q cepstral coeficients (including c0, no liftering).  
      %--- NOTE: N,M,Nband,p,Q must be integers!
      [c,a,P]=PLP(x,N,M,Nband,fs,p,Q);
%        [c,a,P]=RPLP(x,N,M,Nband,fs,p,Q);
%       [c,a,P]=RPLPWavelet(x,N,M,Nband,fs,p,Q);

 % because our PLP model order is larger than the default of 8
 % Append deltas and double-deltas onto the cepstral vectors
%           del = deltas(c);
        % %  Double deltas are deltas applied twice with a shorter window
%           ddel = deltas(deltas(c,5),5);
        % %  Composite, 39-element feature vector, just like we use for speech recognition
%           c = [c;del;ddel]; 
          
%       maximum= max(maximum, length(c(1,:)));


% normalize value of a 13*175 matrix to a range 0~1
%       normC = newc - min(newc(:));
%       newc = normC ./ max(normC(:));
       normC = c - mean(c(:)); 
       c = normC ./ std(normC(:)); %mean=0 ,variance=1
      maximum= 175;  
      %padding extra frames of 0's at the staring side
      padlen = maximum - length(c(1,:));    
      newc =[double(zeros(Q,padlen)) c];
      
      baseFileName=erase(baseFileName, '.wav');
      outputfilename=strcat(ResultFolder,'\',int2str(n),'\', baseFileName,'.txt');
      fileID = fopen(outputfilename,'wt');      
%       I = find(newc > 1);
%taking absolute values of the coefficients
%       newc= abs(newc); 
          



      for ii = 1:Q
        fprintf(fileID,'%f\t',fix(newc(ii,1:end-1)*10^5)/10^5);
        fprintf(fileID,'%f\n',fix(newc(ii,end)*10^5)/10^5);
%       fprintf(fileID,'\r\n');
      end
      
%       copyingStr= strcat('copy ',outputfilename, resultfile);
%       system('copy file.txt r.txt');
      fclose(fileID);
    end
   fclose('all');
end
maximum

% Calculate 12th order PLP features without RASTA
%  [cep2, spec2] = rastaplp(x, fs, 0, 12);
% because our PLP model order is larger than the default of 8
% Append deltas and double-deltas onto the cepstral vectors
%  del = deltas(cep2);
% %  Double deltas are deltas applied twice with a shorter window
%  ddel = deltas(deltas(cep2,5),5);
% %  Composite, 39-element feature vector, just like we use for speech recognition
%  cepDpDD = [cep2;del;ddel];