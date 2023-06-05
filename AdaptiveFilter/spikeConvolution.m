%% function summary: Convolution of spike-like signals with a kernal


function convolved = spikeConvolution(spike, Kernel)

L = length(spike);

positiveSpike = max(spike,0);
positivecConvolved = conv(positiveSpike, Kernel);
positivecConvolved = positivecConvolved(1:L);

% only remains positive ones

convolved = positivecConvolved;

end