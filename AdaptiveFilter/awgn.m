function [y,noisePower] = awgn(sig,reqSNR,varargin)
%AWGN Add white Gaussian noise to a signal.
%   Y = AWGN(X,SNR) adds white Gaussian noise to X.  The SNR is in dB.
%   The power of X is assumed to be 0 dBW.  If X is complex, then 
%   AWGN adds complex noise.
%
%   Y = AWGN(X,SNR,SIGPOWER) when SIGPOWER is numeric, it represents 
%   the signal power in dBW. When SIGPOWER is 'measured', AWGN measures
%   the signal power before adding noise.
%
%   Y = AWGN(X,SNR,SIGPOWER,S) uses S to generate random noise samples with
%   the RANDN function. S can be a random number stream specified by
%   RandStream. S can also be an integer, which seeds a random number
%   stream inside the AWGN function. If you want to generate repeatable
%   noise samples, then either reset the random stream input before calling
%   AWGN or use the same seed input.
%
%   Y = AWGN(..., POWERTYPE) specifies the units of SNR and SIGPOWER. To
%   specify POWERTYPE, both SNR and SIGPOWER must be specified. POWERTYPE
%   can be 'db' or 'linear'. If POWERTYPE is 'db', then SNR is measured in
%   dB and SIGPOWER is measured in dBW.  If POWERTYPE is 'linear', then SNR
%   is measured as a ratio and SIGPOWER is measured in watts assuming 1 ohm
%   reference load.
%
%   [Y,VAR] = AWGN(...) returns the total noise variance used to generate
%   random noise samples.
%
%   Example 1: 
%        % To specify the power of X to be 0 dBW and add noise to produce
%        % an SNR of 10dB, use:
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,10,0);
%
%   Example 2: 
%        % To specify the power of X to be 3 watts and add noise to
%        % produce a linear SNR of 4, use:
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,4,3,'linear');
%
%   Example 3: 
%        % To cause AWGN to measure the power of X and add noise to
%        % produce a linear SNR of 4, use:
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,4,'measured','linear');
%
%   Example 4: 
%        % To specify the power of X to be 0 dBW, add noise to produce
%        % an SNR of 10dB, and utilize a local random stream, use:
%        S = RandStream('mt19937ar','Seed',5489);
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,10,0,S);
%
%   Example 5: 
%        % To specify the power of X to be 0 dBW, add noise to produce
%        % an SNR of 10dB, and produce reproducible results, use:
%        reset(RandStream.getGlobalStream)
%        X = sqrt(2)*sin(0:pi/8:6*pi);
%        Y = awgn(X,10,0);
%
%
%   See also convertSNR, WGN, RANDN, RandStream/RANDN, BSC.

%   Copyright 1996-2021 The MathWorks, Inc.

%#codegen

narginchk(2,5);

% Validate signal input
validateattributes(sig, {'numeric'}, ...
    {'nonempty'}, 'awgn', 'signal input');

% Validate SNR input
validateattributes(reqSNR, {'numeric'}, ...
    {'real','scalar','nonempty'}, 'awgn', 'SNR input');

% Validate signal power
if nargin >= 3
    if strcmpi(varargin{1}, 'measured')
       sigPower = sum(abs(sig(:)).^2)/numel(sig); % linear
    else
        validateattributes(varargin{1}, {'numeric'}, ...
            {'real','scalar','nonempty'}, 'awgn', 'signal power input');
        sigPower = varargin{1}; % linear or dB
    end
else
    sigPower = 1; % linear, default
end

% Validate state or power type
if nargin >= 4    
    coder.internal.errorIf(comm.internal.utilities.isCharOrStringScalar(varargin{2}) && ...
        all(~strcmpi(varargin{2}, {'db','linear'})), ...
        'comm:awgn:InvalidPowerType');
    
    isStream = ~isempty(varargin{2}) && ~comm.internal.utilities.isCharOrStringScalar(varargin{2});
    
    if isStream && ~isa(varargin{2}, 'RandStream') % Random stream seed
        validateattributes(varargin{2}, {'double'}, ...
            {'real','scalar','nonnegative','integer','<',2^32}, ...
            'awgn', 'seed input');
    end
else % Default
    isStream = false;
end

% Validate power type
if nargin == 5
    coder.internal.errorIf(comm.internal.utilities.isCharOrStringScalar(varargin{2}), ... % Type has been specified as the 4th input
        'comm:awgn:InputAfterPowerType'); 
    coder.internal.errorIf(all(~strcmpi(varargin{3}, {'db','linear'})), ...
        'comm:awgn:InvalidPowerType'); 
end

isLinearScale = ((nargin == 4) && ~isStream && strcmpi(varargin{2}, 'linear')) || ...
    ((nargin == 5) && strcmpi(varargin{3}, 'linear'));

% Cross-validation
coder.internal.errorIf(isLinearScale && (sigPower < 0), ...
    'comm:awgn:InvalidSigPowerForLinearMode');

coder.internal.errorIf(isLinearScale && (reqSNR < 0), ...
    'comm:awgn:InvalidSNRForLinearMode');

if ~isLinearScale  % Convert signal power and SNR to linear scale
    if (nargin >= 3) && ~comm.internal.utilities.isCharOrStringScalar(varargin{1}) % User-specified signal power
        sigPower = 10^(sigPower/10);
    end
    reqSNR = 10^(reqSNR/10);
end

noisePower = sigPower/reqSNR;

if isStream
    if isa(varargin{2}, 'RandStream')
        stream = varargin{2};
    elseif isempty(coder.target)
        stream = RandStream('shr3cong', 'Seed', varargin{2});
    else        
        stream = coder.internal.RandStream('shr3cong', 'Seed', varargin{2});
    end
    
    if ~isreal(sig)
        noise = sqrt(noisePower/2)* complex(randn(stream, size(sig)), ...
                                  randn(stream, size(sig)));
    else
        noise = sqrt(noisePower)* randn(stream, size(sig));
    end
else
    if ~isreal(sig)
        noise = sqrt(noisePower/2)* complex(randn(size(sig)), randn(size(sig)));
    else
        noise = sqrt(noisePower)* randn(size(sig));
    end
end    

y = sig + noise; 