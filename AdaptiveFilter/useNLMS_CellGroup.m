function [AF, es6, y]=useNLMS_CellGroup(sigR,sigG, extendfactor)

sigG = newDenoise(sigG);

sigR = newDenoise(sigR);

% extend the signal
L = length(sigG);

MM = prctile(sigG,10);
prefixSig(1:ceil(L*extendfactor)) = MM;
prefixSig = awgn(prefixSig,1);
sigG = [prefixSig sigG];

MM1 = prctile(sigR,10);
prefixSig1(1:ceil(L*extendfactor)) = MM1;
prefixSig1 = awgn(prefixSig1,1);
sigR = [prefixSig1 sigR];


[e,y,~] = myNLMS_t(sigR, sigG, 0.5, 2, -1e-3, -1e-4);

% % % e = thresholdDenoise3(e, mean(sigG)*0.1,0.05);
e = e(ceil(L*extendfactor)+1:end);
y = y(ceil(L*extendfactor)+1:end);
y(y==0) = sigG(y==0);
es6 = movsum(e,6);
es6 = thresholdDenoise3(es6,mean(sigG)*0.4,0.05);
es6 = es6./y;

ARKernel = generateAR1(0.95);
AF = spikeConvolution(es6/8, ARKernel);

end