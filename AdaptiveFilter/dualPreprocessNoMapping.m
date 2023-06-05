%% function summary: an preprocessing of the raw dual channel signals
% Wrote by Chen Shen

%  input:
%   sigRraw --- raw RED signal 
%   sigGraw --- raw GREEN signal

% output: 
%   nsigR --- the denoised RED signal
%   nsigG --- the denoised GREEN signal       
function [nsigR, nsigG] = dualPreprocessNoMapping(sigRraw, sigGraw)


% linear re-mapping red channel signal to the green one (cancelled)
% because it is no difference when using Adaptive Filter
% sigR = mapminmax(sigRraw,min(sigGraw),max(sigGraw));
% sigG = sigGraw;

sigR = sigRraw;
sigG = sigGraw;

% build low-pass filter
[b, a] = butter(5,0.3,'low');
% denoise 30% high-frequency component
nsigR=filtfilt(b,a,sigR);
nsigG=filtfilt(b,a,sigG);

% nsigR = waveletDenoise(nsigR);
% nsigG = waveletDenoise(nsigG);
% Channel Vector Correction Algorithm %
% number of fft
% Nfft=2^ceil(log2(L));
% 
% sf1 = fft(sigR,Nfft);
% sf2 = fft(sigG,Nfft);
% 
% % find the compensation vector v in the complex space
% v = max(sf1(1:Nfft/2))./max(sf2(1:Nfft/2));
% sf2 = [sf2(1:Nfft/2).*v sf2(Nfft/2+1:end).*conj(v)]; 
% 
% 
% nsigR = real(ifft(sf1,Nfft)); nsigR = nsigR(1:L);
% nsigG = real(ifft(sf2,Nfft)); nsigG = nsigG(1:L);

end