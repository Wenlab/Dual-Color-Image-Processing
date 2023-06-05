%% function summary: an implementation of Generating AR(1) Kernel
% Wrote by Chen Shen

%  input:
%   gamma --- Auto Regression processing coefficient, typically 0.95
%   sig --- the input calcium signal
%   inferredBase --- the inferred baseline by adaptive filter

% output: AR --- the AR kernal for convolution



function AR = generateAR1(gamma, sig, inferredBase)
% Length = 40, 4s decay 
Length = 40;
s = zeros(1,Length);

s(2)=1;

AR=zeros(1,Length);
AR(1)=0;

for t=2:Length

    AR(t)=gamma*AR(t-1)+s(t);

end

factor = mean(sig)/mean(inferredBase);
AR = AR.*factor;


end