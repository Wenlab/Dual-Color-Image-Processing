function nsig = newDenoise(sig)

nsig1=sig;

sDiff=diff(sig);

le=length(sig);

for ii=2:le-1
    if sign(sDiff(ii-1))~=sign(sDiff(ii)) && abs(sig(ii+1)-sig(ii-1))<abs(sDiff(ii-1))/3
        nsig1(ii)=(sig(ii-1)+sig(ii+1))/2;
    end
end

DFoP=diff(nsig1)./nsig1(1:le-1);
AT=find(DFoP>0.08);
AT(AT<3)=[];
AT(AT>le-4)=[];
nsig = smoothdata(nsig1,'gaussian',10);
le2=length(AT);
for jj=1:le2
    nsig(AT(jj)-2:AT(jj)+4)=nsig1(AT(jj)-2:AT(jj)+4);
end

end