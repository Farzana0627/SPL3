function [cepstra, spectra, pspectrum, lpcas, F, M] = rastaplp(samples, sr, dorasta, modelorder)
%[cepstra, spectra, lpcas] = rastaplp(samples, sr, dorasta, modelorder)
%
% cheap version of log rasta with fixed parameters
%
% output is matrix of features, row = feature, col = frame
%
% sr is sampling rate of samples, defaults to 8000
% dorasta defaults to 1; if 0, just calculate PLP
% modelorder is order of PLP model, defaults to 8.  0 -> no PLP
%
% rastaplp(d, sr, 0, 12) is pretty close to the unix command line
% feacalc -dith -delta 0 -ras no -plp 12 -dom cep ...
% except during very quiet areas, where our approach of adding noise
% in the time domain is different from rasta's approach 
%
% 2003-04-12 dpwe@ee.columbia.edu after shire@icsi.berkeley.edu's version

if nargin < 2
  sr = 8000;
end
if nargin < 3
  dorasta = 1;
end
if nargin < 4
  modelorder = 8;
end
fbtype='mel';
% add miniscule amount of noise
%samples = samples + randn(size(samples))*0.0001;

% first compute power spectrum
[pspectrum, e, nfft]  = powspec(samples, sr);

% next group to critical bands
nbands = 257;
aspectrum = audspec(pspectrum, sr,nbands,fbtype);
nbands = size(aspectrum,1);

if dorasta ~= 0

  % put in log domain
  nl_aspectrum = log(aspectrum);

  % next do rasta filtering
  ras_nl_aspectrum = rastafilt(nl_aspectrum);

  % do inverse log
  aspectrum = exp(ras_nl_aspectrum);

end
  
% do final auditory compressions
postspectrum = postaud(aspectrum, sr/2, fbtype); % 2012-09-03 bug: was sr

if modelorder > 0

  % LPC analysis 
  lpcas = dolpc(postspectrum, modelorder);

  % convert lpc to cepstra
  cepstra = lpc2cep(lpcas, modelorder+1);

  % .. or to spectra
  [spectra,F,M] = lpc2spec(lpcas, nbands);

else
  
  % No LPC smoothing of spectrum
  spectra = postspectrum;
%   cepstra = spec2cep(spectra);
  cepstra=spectra;
end
% X=cepstra;
% Nframes=length(X(1,:)); %number of frames
% Nlevels=log2(length(X(:,1))); %number of dwt levels
% for i=1:Nframes
%     lowpass=X(:,i);
%     prev=[];
%     coefficients=[];
%     for j=1:Nlevels      
%         [a ,d]= dwt(lowpass, 'haar');
%         lowpass=a;    
%         coefficients=cat(1,prev,d);%added a=coefficients of approximation, d=coefficients of detail
%         prev=coefficients;
%     %         coefficients=abs(coefficients);%added
%     end
%     dwtX(:,i)= fliplr(coefficients);
% end
% 
% cepstra=dwtX(:,:).^2;

% Daubechies'db1' or 'haar', 'db2', ... ,'db10', ... , 'db45' 
% Coiflets 'coif1', ... , 'coif5'
% [Lo_D,Hi_D] = wfilters('coif3','d');  

% Hi_D and Hi_R (High-pass filters) If 'type' = 'h'
% Lo_D and Lo_R	(Low-pass filters) If 'type' = 'l'
% Lo_R and Hi_R	(Reconstruction filters) If 'type' = 'r'
% Lo_D and Hi_D	(Decomposition filters)If 'type' = 'd'
[r c]= size(cepstra);
L = wmaxlev(r,'db2');
for i=1:c
    [cA,cD] = wavedec(cepstra(:,i),L,'db2'); 
    ynew(:,i)=cA;
end
cepstra=ynew;
% cepstra = spec2cep(spectra);
cepstra = lifter(cepstra, 0.6);
