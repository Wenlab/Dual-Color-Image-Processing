function nsig = newDenoise(sig)

nsig=sig;

sDiff=diff(sig);

le=length(sig);

for ii=2:le-1
    if sign(sDiff(ii-1))~=sign(sDiff(ii)) && abs(sig(ii+1)-sig(ii-1))<abs(sDiff(ii-1))/3
        nsig(ii)=(sig(ii-1)+sig(ii+1))/2;
    end
end

end