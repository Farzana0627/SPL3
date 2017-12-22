addpath('rastamat');
addpath('AuditoryToolbox\NewOnes');

% myFolder = 'G:\8th semester dropbox extras\ThesisDataset\Test_set';
myFolder = 'G:\8th semester dropbox extras\ThesisDataset\Train_set';
% ResultFolder= 'RPLPWaveletTrainResultFolder';
% ResultFolder= 'RPLPTrainResultFolder';
% ResultFolder= 'PLPTrainResultFolder';
% ResultFolder= 'RPLPWaveletModifiedTrainResultFolder';
ResultFolder= 'RPLPWaveletModified3TrainResultFolder';

% ResultFolder= 'RPLPWaveletTestResultFolder';
% ResultFolder= 'RPLPTestResultFolder';
% ResultFolder= 'PLPTestResultFolder';
% ResultFolder= 'RPLPWaveletModifiedTestResultFolder';
% ResultFolder= 'RPLPWaveletModified3TestResultFolder';

% Check to make sure that folder actually exists.  Warn user if it doesn't
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
if ~isdir(ResultFolder)
   mkdir (ResultFolder);
end
maximum=0;
 maxi=0;
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
      % x: signal
      % fs: sampling frequency
      [x,fs] = audioread(fullFileName);
      [x,fs]= Preprocessing(x, fs, n, k);
      x= Preemphasis(x);
      % N: frame length 
      N= 400;
      % M: frame rate in samples here 10 ms 
      M= 160;
      % Nband: num. of (effective) critical bands
      Nband=257;
      % P: auditory power spectrum (Nband+2)x(Nframes) : 
      p=5; %hermansky (5th order) max= 15, if PLP model order is larger than the default of 8, append deltas and double-deltas onto the cepstral vectors
      Q=39; %Q cepstral coeficients for each frame
      [cep2, spec2] = rastaplp(x,fs, 0, 12);
       % because our PLP model order is larger than the default of 8
       % Append deltas and double-deltas onto the cepstral vectors
       del = deltas(cep2);
       % Double deltas are deltas applied twice with a shorter window
       ddel = deltas(deltas(cep2,5),5);
       % Composite, 39-element feature vector, just like we use for speech recognition
       c = [cep2;del;ddel]; 
       maxi= max(maxi, length(c(1,:)));
% normalize value to a range 0~1
%       normC = newc - min(newc(:));
%       newc = normC ./ max(normC(:));
      normC = c - mean(c(:)); 
      c = normC ./ std(normC(:)); %mean=0 ,variance=1
      maximum= 32;  
      %padding extra frames of 0's at the staring side
      padlen = maximum - length(c(1,:));    
      newc =[double(zeros(Q,padlen)) c];
      
      baseFileName=erase(baseFileName, '.wav');
      outputfilename=strcat(ResultFolder,'\',int2str(n),'\', baseFileName,'.txt');
      fileID = fopen(outputfilename,'wt');      
      for ii = 1:Q
        fprintf(fileID,'%f\t',fix(newc(ii,1:end-1)*10^5)/10^5);
        fprintf(fileID,'%f\n',fix(newc(ii,end)*10^5)/10^5);
      end
      fclose(fileID);
    end
   fclose('all');
end
maxi
