
addpath('rastamat');
[x fs]= audioread('410.wav');
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);

%mono channeling
[r,c]= size(x);
if c>1

    sum=zeros([r,1]);
    for j=1:c          
        sum=sum+x(:,j);
    end
    x=sum/c;        
end
x=double(x);

%preemphasis
B=[1 -.95];

x= filter(B,1,x);
t=linspace(0,1000,numel(x));
% figure('PaperPositionMode', 'auto');
% plot(t, x, 'b-', 'LineWidth',.01);
% title('Signal X(t) in the Time Domain after Pre-Emphasis')
% xlabel('Time in miliseconds')
% ylabel('Amplitude')
% print(gcf, '-dpdf', 'preemphasis.pdf');

%sampling
[P,Q] = rat(16e3/fs);
% Sample at 16000 Hz
x = resample(x,P,Q);
fs=16e3;
% t=linspace(0,1000,numel(x));
% figure('PaperPositionMode', 'auto');
% plot(t, x, 'b-', 'LineWidth',.01);
% title('Signal X(t) sampled at 16kHz')
% xlabel('Time in miliseconds')
% ylabel('Amplitude')
% print(gcf, '-dpdf', 'sampled.pdf');

%windowing
winvec = hann(length(x));
x=x.*winvec;
% t=linspace(0,1000,numel(x));
% figure('PaperPositionMode', 'auto');
% plot(t, x, 'b-', 'LineWidth',.01);
% title('Windowed signal using Hanning function')
% xlabel('Time in miliseconds')
% ylabel('Amplitude')
% print(gcf, '-dpdf', 'windowed.pdf');

%fft
% xdft = fft(x);
% y=abs(xdft).^2;
% y = y + 1;
% L= length(Y);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = fs*(0:(L/2))/L;
% figure('PaperPositionMode', 'auto');
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum P1 of windowed X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% print(gcf, '-dpdf', 'fft.pdf');
%framing
wintime = 0.025;
steptime = 0.010;
winpts = round(wintime*fs);
steppts = round(steptime*fs);
nfft = 2^(ceil(log(winpts)/log(2)));
window = [hanning(winpts)'];
noverlap = winpts - steppts;
SAMPRATE = fs;
x = abs(myspecgram(x*32768,nfft,SAMPRATE,window,noverlap)).^2; %powerspec
% x = x + winpts;

%Filter Banks

[nfreqs,nframes] = size(x);
nfft = (nfreqs-1)*2;
nfilts=10;
bwidth=1;
minfreq=0;
maxfreq=8000;
% wts = fft2melmx(nfft, SAMPRATE, nfilts, bwidth, minfreq, maxfreq);
wts = fft2barkmx(nfft, SAMPRATE, nfilts, bwidth, minfreq, maxfreq);
wts = wts(:, 1:nfreqs);

aspectrum = wts * x;
f=linspace(minfreq,maxfreq,257);
plot(f,wts);
% title('Mel Filter banks of the first frame of digit 4 (char)')
% xlabel('f (Hz)')
% ylabel('Amplitude')
% print(gcf, '-dpdf', 'melfilt.pdf');
title('Bark Filter banks of the first frame of digit 4 (char)')
xlabel('f (Hz)')
ylabel('Amplitude')
print(gcf, '-dpdf', 'Barkfilt.pdf');
[phi,psi,xval] = wavefun('coif5',10);
subplot(211);
plot(xval,phi);
title('coif5 Scaling Function');
subplot(212);
plot(xval,psi);
title('coif5 Wavelet');
