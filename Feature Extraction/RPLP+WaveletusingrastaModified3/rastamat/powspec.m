function [y,e,NFFT] = powspec(x, sr, wintime, steptime, dither)
%[y,e] = powspec(x, sr, wintime, steptime, sumlin, dither)
%
% compute the powerspectrum and frame energy of the input signal.
% basically outputs a power spectrogram
%
% each column represents a power spectrum for a given frame
% each row represents a frequency
%
% default values:
% sr = 8000Hz
% wintime = 25ms (200 samps)
% steptime = 10ms (80 samps)
% which means use 256 point fft
% hamming window
%
% $Header: /Users/dpwe/matlab/rastamat/RCS/powspec.m,v 1.3 2012/09/03 14:02:01 dpwe Exp dpwe $

% for sr = 8000
%NFFT = 256;
%NOVERLAP = 120;
%SAMPRATE = 8000;
%WINDOW = hamming(200);

if nargin < 2
  sr = 8000;
end
if nargin < 3
  wintime = 0.032;
end
if nargin < 4
  steptime = 0.020;
end
if nargin < 5
  dither = 1;
end

winpts = round(wintime*sr);
steppts = round(steptime*sr);

NFFT = 2^(ceil(log(winpts)/log(2)));
%WINDOW = hamming(winpts);
%WINDOW = [0,hanning(winpts)'];
WINDOW = [hanning(winpts)'];
% hanning gives much less noisy sidelobes
NOVERLAP = winpts - steppts;
SAMPRATE = sr;

% Values coming out of rasta treat samples as integers, 
% not range -1..1, hence scale up here to match (approx)
y = abs(myspecgram(x*32768,NFFT,SAMPRATE,WINDOW,NOVERLAP)).^2;

% y = fft(x,NFFT).^2;
% imagine we had random dither that had a variance of 1 sample 
% step and a white spectrum.  That's like (in expectation, anyway)
% adding a constant value to every bin (to avoid digital zero)
if (dither)
  y = y + winpts;
end
% Daubechies'db1' or 'haar', 'db2', ... ,'db10', ... , 'db45' 
% Coiflets 'coif1', ... , 'coif5'
% [Lo_D,Hi_D] = wfilters('coif3','d');  

% Hi_D and Hi_R (High-pass filters) If 'type' = 'h'
% Lo_D and Lo_R	(Low-pass filters) If 'type' = 'l'
% Lo_R and Hi_R	(Reconstruction filters) If 'type' = 'r'
% Lo_D and Hi_D	(Decomposition filters)If 'type' = 'd'
[r c]= size(y);
L = wmaxlev(r,'db32');
% ynew= zeros(382,c);
for i=1:c
    [cA,cD] = wavedec(y(:,i),L,'db32'); 
    ynew(:,i)=cA;
end
y=ynew;
% [cA,cD] = dwt2(x,'db32');
% y=abs(cD).^2;
% if (dither)
%   y = y + winpts;
% end
% ignoring the hamming window, total power would be = #pts
% I think this doesn't quite make sense, but it's what rasta/powspec.c does

% that's all she wrote

% 2012-09-03 Calculate log energy - after windowing, by parseval
e = log(sum(y));
