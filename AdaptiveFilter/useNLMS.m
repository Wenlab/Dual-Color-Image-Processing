function [AF, e, y]=useNLMS_2(sigR,sigG, extendfactor)

sigG = newDenoise(sigG);
sigR = newDenoise(sigR);
% extend the signal
L = length(sigG);

MM = mean(sigG);
prefixSig(1:ceil(L*extendfactor)) = MM;
prefixSig = awgn(prefixSig,1);
sigG = [prefixSig sigG];

MM1 = mean(sigR);
prefixSig1(1:ceil(L*extendfactor)) = MM1;
prefixSig1 = awgn(prefixSig1,1);
sigR = [prefixSig1 sigR];


[e,y,~] = myNLMS(sigR, sigG, 0.8, 2, -1e-3, -1e-4);

e = thresholdDenoise3(e, mean(sigG)*0.1,0.05);
e = e(ceil(L*extendfactor)+1:end);
y = y(ceil(L*extendfactor)+1:end);
e = e./y;

ARKernel = generateAR1(0.95);
AF = spikeConvolution(e, ARKernel);

end