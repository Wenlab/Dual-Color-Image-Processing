%% function summary: To Call the NLMS algorithm with parameters

%  input:
%   sigR --- the red channel signal
%   sigG --- the green channel signal
%   extendfactor --- Extend the length (percentage) of the signal 
%  output:
%   AF --- the AF signal
%   e --- the output error vector
%   y --- the inferred adaptive baseline

function [AF, e, y] = useNLMS(sigR,sigG, extendfactor)

% Preprocessing
[nsigR, nsigG] = dualPreprocessNoMapping(sigR, sigG);

% extend the signal for convergence
% assumed long and stationary signal,
% and absence of neural activity in the beginning
L = length(nsigG);
nsigR = padarray(nsigR, [0 ceil(L*extendfactor)],'symmetric','pre');
nsigG = padarray(nsigG, [0 ceil(L*extendfactor)],'symmetric','pre');

% alternatively, it can also add the Baseline with white noise to the front
% as follow:

% BaseLine = mean(nsigG(end-L*(1-extendfactor):end));
% prefixSig(1:L*extendfactor) = BaseLine;
% prefixSig = awgn(prefixSig,snr(nsigG)+2);
% nsigG(1:L*extendfactor) = prefixSig;
% 
% BaseLine = mean(nsigR(end-L*(1-extendfactor):end));
% prefixSig(1:L*extendfactor) = BaseLine;
% prefixSig = awgn(prefixSig,snr(nsigR)+2);
% nsigR(1:L*extendfactor) = prefixSig;

% call the NLMS algorithm
[e ,y, w] = myNLMS(nsigR, nsigG, 0.8, 2, -1e-3, -1e-4);

 % build low-pass filter
[b, a] = butter(3,[0.1 0.35],'bandpass');

% denoise 30% high-frequency component
e = filtfilt(b,a,e);
e = thresholdDenoise3(e, mean(nsigG)*0.05,0.05);
e = e(ceil(L*extendfactor)+1:end);
% waveletDenoise
e=waveletDenoise(e);

y = y(ceil(L*extendfactor)+1:end);

% generate AR(1) kernel for convolution
ARKernel = generateAR1(0.95,nsigG,y);

AF = spikeConvolution(e, ARKernel);


end