%% output: Coherence3.mat: 
%%         A3 (#pixels x #regions): footprint of each region (0 or 1)
%%         CalTrace3_original (#regions x # time points): raw calcium traces
%%         Coherence3 (same size as the volume): coherence (the correlation coefficient between a pixel and the region it 
%%           belongs to) of each region after thresholding
%%         L_temp3 (same size as the volume): footprint of each region (index of the region that the pixel belongs to, 
%%           or 0 for background)


clear all;
close all;
%%
file_path =  'G:\\calcium_image\\working\\200705\\*';
FoldInfo=dir(file_path);
A=[];
for i=1:size(FoldInfo,1)
    if FoldInfo(i).isdir==0
        A=[A,i];
    end
end
FoldInfo(A)=[];
FoldInfo(1:2)=[]; % exclude '.' and '..'
stepSize = 1;

file_path1= 'G:\\calcium_image\\working\\200705\\';
file_path2= '\\';

% % % e=size(FoldInfo,1);

for ii=1:size(FoldInfo,1)
    B=FoldInfo(ii).name;
    path(ii,:)=[file_path1,B,file_path2];
end

b=size(path,1);

%% segment by CorrMap
%% parameters
% % % clear;
% thresh_Var = 0.1;
thresh_Fmax = 9;
thresh_Fmin = 2;
% min_size = 27;
r = 1;
SE = strel('sphere',r); % for imopen
adjacent_distance = 3;
adjacent_distance2 = 2; % sqrt(adjacent_distance^2/3)
% % % file_path = 'D:\\cygwin64\\home\\USER\\20_01_08_05\\';  % need to change
file_path4 = path(1,:);
input_filename = 'affine_noEyes';
input_extend = '.mat';
% % % input_filename = 'affine'; % need to change
% % % input_extend = '.nii'; % need to change
% % % frame_start = 1;% need to change
% % % frame_end = 200;% need to change
filename_in = [input_filename,num2str(1),input_extend];
load([file_path4,filename_in]);
% % % filename_in = [input_filename,num2str(frame_start),input_extend];
% % % ObjRecon = niftiread([file_path,filename_in]);% need to change
dx = size(ObjRecon,1);
dy = size(ObjRecon,2);
dz = size(ObjRecon,3);

%%
Corr = zeros(dx,dy,dz);
SD = zeros(dx,dy,dz);
Y_mean = zeros(dx,dy,dz);
F_max = zeros(dx,dy,dz);
F_min = ObjRecon;
% DFoF_max = zeros(dx,dy,dz);

%% first round: calculate Y_mean, F_max, F_min
% % % Y_mean = Y_mean + ObjRecon;% need to change
file_path3= 'affine_noEyes*';
d=zeros(b,1);
g=zeros(b,1);
for m=1:b
    path2=[path(m,:),file_path3];
    
    FoldInfo1=dir(path2);
    c=size(FoldInfo1,1);
    d(m)=c;
    for f=1:c
        tic
        ff = (f - 1)*stepSize + 1;
        filename_in = [input_filename,num2str(ff),input_extend];
% % %         ObjRecon = niftiread([file_path,filename_in]);% need to change
        load([path(m,:),filename_in]);
        Y_mean = Y_mean + ObjRecon;
        clear temp;
        temp(:,:,:,1) = ObjRecon;
        temp(:,:,:,2) = F_max;
        F_max = squeeze(max(temp,[],4));
        temp(:,:,:,2) = F_min;
        F_min = squeeze(min(temp,[],4));
        disp mean
        disp(m)
        disp(ff)
        toc;
    end
end
Y_mean = Y_mean/sum(d);% need to change
Y_mean(F_max<thresh_Fmax) = 0; % mask by thresh_Fmax
Y_mean(F_min<thresh_Fmin) = 0; % mask by thresh_Fmin

for m=2:b
   g(m) = sum(d(1:m-1));
end

%% second round: calculate SD,
for m=1:b
    for f=1:d(m)
        tic;
        ff = (f - 1)*stepSize + 1;
        filename_in = [input_filename,num2str(ff),input_extend];
        % % %     ObjRecon = niftiread([file_path,filename_in]);% need to change
        load([path(m,:),filename_in]);
        Y_shift = bsxfun(@minus,ObjRecon,Y_mean);
        SD = SD + Y_shift.*Y_shift;
        disp SD
        disp(m)
        disp(ff)
        toc;
    end
end
SD = sqrt(SD/sum(d));% need to change
SD(F_max<thresh_Fmax) = 0; % mask by thresh_Fmax
SD(F_min<thresh_Fmin) = 0; % mask by thresh_Fmin




%% third round: calculate Corr,
for m=1:b
    for f=1:d(m)
        tic;
        ff = (f - 1)*stepSize + 1;
        % % % for f=frame_start:frame_end% need to change
        filename_in = [input_filename,num2str(ff),input_extend];
% % %         ObjRecon = niftiread([file_path,filename_in]);% need to change
        load([path(m,:),filename_in]);
        Y_shift = bsxfun(@minus,ObjRecon,Y_mean);
        Y_shift = Y_shift./SD;
        Y_shift(SD==0) = 0;
        Y_shift(F_max<thresh_Fmax) = 0; % mask by thresh_Fmax
        Y_shift(F_min<thresh_Fmin) = 0; % mask by thresh_Fmin
        for i=1+adjacent_distance:dx-adjacent_distance
            for j=1+adjacent_distance:dy-adjacent_distance
                for k=1+adjacent_distance:dz-adjacent_distance
                    % c1 = Y_shift(i,j,k)*Y_shift(i+1,j,k);
                    % c2 = Y_shift(i,j,k)*Y_shift(i,j+1,k);
                    % c3 = Y_shift(i,j,k)*Y_shift(i,j,k+1);
                    % c4 = Y_shift(i,j,k)*Y_shift(i-1,j,k);
                    % c5 = Y_shift(i,j,k)*Y_shift(i,j-1,k);
                    % c6 = Y_shift(i,j,k)*Y_shift(i,j,k-1);
                    % Cov(i,j,k) = Cov(i,j,k)+c1+c2+c3+c4+c5+c6;
                    % Cov(i,j,k) = Cov(i,j,k) + (Y_shift(i+1,j,k)+Y_shift(i,j+1,k)+Y_shift(i,j,k+1)+Y_shift(i-1,j,k)+Y_shift(i,j-1,k)+Y_shift(i,j,k-1))*Y_shift(i,j,k);
                    % adjacent_points = computeAdjacentPoints(i,j,k,adjacent_distance,dx,dy,dz);
                    % num_adjacent_points = size(adjacent_points,1);
                    % adjacent_points = [i-adjacent_distance, j, k; i+adjacent_distance, j, k; i, j-adjacent_distance, k; i, j+adjacent_distance, k; i, j, k-adjacent_distance; i, j, k+adjacent_distance];
                    temp = Y_shift(i-adjacent_distance, j, k) + Y_shift(i+adjacent_distance, j, k) + Y_shift(i, j-adjacent_distance, k) + Y_shift(i, j+adjacent_distance, k) + Y_shift(i, j, k-adjacent_distance) + Y_shift(i, j, k+adjacent_distance);
                    temp = temp + Y_shift(i+adjacent_distance2, j+adjacent_distance2, k+adjacent_distance2) + Y_shift(i-adjacent_distance2, j+adjacent_distance2, k+adjacent_distance2);
                    temp = temp + Y_shift(i+adjacent_distance2, j-adjacent_distance2, k+adjacent_distance2) + Y_shift(i-adjacent_distance2, j-adjacent_distance2, k+adjacent_distance2);
                    temp = temp + Y_shift(i+adjacent_distance2, j+adjacent_distance2, k-adjacent_distance2) + Y_shift(i-adjacent_distance2, j+adjacent_distance2, k-adjacent_distance2);
                    temp = temp + Y_shift(i+adjacent_distance2, j-adjacent_distance2, k-adjacent_distance2) + Y_shift(i-adjacent_distance2, j-adjacent_distance2, k-adjacent_distance2);
                    Corr(i,j,k) = Corr(i,j,k) + temp*Y_shift(i,j,k);
                end
            end
        end
        % DFoF = Y_shift./Y_mean;
        % clear temp;
        % temp(:,:,:,1) = DFoF;
        % temp(:,:,:,2) = DFoF_max;
        % DFoF_max = squeeze(max(temp,[],4));
        % f
        disp Corr
        disp(m)
        disp(ff)
        toc;
    end
end
% Cov = Cov/(frame_end-frame_start+1);
Corr = Corr/sum(d)/14;% need to change
Corr_original = Corr;
clear Y_shift;
clear ObjRecon;

% min_Cov = min(Cov(:));
% max_Cov = max(Cov(:));
% Cov = (Cov-min_Cov)/(max_Cov-min_Cov);
min_Corr = min(Corr(:));
max_Corr = max(Corr(:));
Corr = (Corr-min_Corr)/(max_Corr-min_Corr);

% L = watershed(1-Cov);
L = watershed(1-Corr);

L_temp2 = L;
num_components = max(L(:));
% L_temp(Cov<thresh_Cov) = 0;
L_temp2(F_max<thresh_Fmax) = 0;
L_temp2(F_min<thresh_Fmin) = 0;
% L_temp2 = bwareaopen(L_temp2,min_size,6);
L_temp2 = imopen(L_temp2,SE); % To exclude small or thin areas and spines, bridges, etc.
L_temp2 = uint16(bwlabeln(L_temp2,6));
num_components_keep = max(L_temp2(:));
A = sparse(dx*dy*dz,1);
for k=1:num_components_keep
    temp  = (L_temp2==k);
    A(:,k) = sparse(reshape(double(temp),[dx*dy*dz,1]));
end

%% save A and L_temp2
Fmean = Y_mean;
save([file_path1,'CorrMap.mat'],'A','L_temp2','L','SD','F_max','F_min','Fmean','Corr_original');

%% display
index_slice = 1:dz;
num_slices = length(index_slice);
cmap = parula;

max_Corr = max(Corr(:));

clear slices;
slices(:,:,1:num_slices) = Corr(:,:,index_slice);
clear picture;
clear RGB;
for l=1:num_slices
    picture(:,:,l) = uint8(slices(:,:,l)/max_Corr*63);
    RGB(:,:,:,l) = ind2rgb(squeeze(picture(:,:,l)),cmap);
    RGB(:,:,1,l) = RGB(:,:,1,l);
    RGB(:,:,2,l) = RGB(:,:,2,l);
    RGB(:,:,3,l) = RGB(:,:,3,l);
    % figure(l);hold off;imshow(picture);hold on;
end

imwrite(squeeze(RGB(:,:,:,1)),[file_path1,'CorrMap.tif']);
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),[file_path1,'CorrMap.tif'],'WriteMode','append');
end
clear slices;
clear picture;
clear RGB;

temp = uint16(L);
imwrite(squeeze(temp(:,:,1)),[file_path1,'L.tif']);
for l=2:num_slices
    imwrite(squeeze(temp(:,:,l)),[file_path1,'L.tif'],'WriteMode','append');
end

temp = uint16(L_temp2);
imwrite(squeeze(temp(:,:,1)),[file_path1,'L_temp2.tif']);
for l=2:num_slices
    imwrite(squeeze(temp(:,:,l)),[file_path1,'L_temp2.tif'],'WriteMode','append');
end

clear Corr;
clear L;
clear F_max;
clear F_min;
clear Corr_original;
clear temp;
clear Y_mean;

%% extract Calcium traces
%% set parameters
% input: A
% output: CalTrace
K = size(A,2);
e = size(A,1);
% % % T = frame_end - frame_start + 1;
T = sum(d);
batch_size = 1;

%%
CalTrace = zeros(K,T);

for m=1:b
    for ff=1:batch_size:d(m)
        tic;
        % % % for ff=frame_start:batch_size:frame_end
        Y_r = zeros(e,min([d(m),ff+batch_size-1])-ff+1); % load a batch
        for f=ff:min([d(m),ff+batch_size-1])
% % %             ObjRecon = niftiread([file_path,input_filename,num2str(f),input_extend]);
        fff = (f - 1)*stepSize + 1;
        filename_in = [input_filename,num2str(fff),input_extend];
        load([path(m,:),filename_in]);
            Y_r(:,f-ff+1) = reshape(ObjRecon,[e,1]);
        end
        for k=1:K
            temp = Y_r(A(:,k)>0,:);
% % %             CalTrace(k,ff:min([d(m),ff+batch_size-1])) = mean(temp,1);
            CalTrace(k,g(m)+ff:g(m)+min([d(m),ff+batch_size-1])) = mean(temp,1);
        end
        disp CalciumTrace
        disp(m)
        disp(fff)
        toc;
    end
    
    
end
save([file_path1,'CalTrace.mat'],'CalTrace');

clear temp;
clear Y_r;

%% calculate the coherence map
%% input: CalTrace, L_temp2, Fmean, SD
%% set parameters
d1 = dx;
d2 = dy;
d3 = dz;

%%
% % % T = frame_end - frame_start + 1;
T = sum(d);
t_step = 1;
% % % ObjRecon = niftiread([file_path,input_filename,num2str(1),input_extend]);
filename_in = [input_filename,num2str(1),input_extend];
load([path(1,:),filename_in]);

%% normalize CalTrace
CalTrace = CalTrace - mean(CalTrace,2);
SD_CalTrace = sqrt(sum(CalTrace.*CalTrace,2)/T);
SD_CalTrace = repmat(SD_CalTrace, 1,T);
CalTrace = CalTrace./SD_CalTrace;

Coherence = zeros(size(ObjRecon));
for m=1:b
    for t=1:t_step:d(m)
        tic;
        ff = (t - 1)*stepSize + 1;
        % % %     ObjRecon = niftiread([file_path,input_filename,num2str(t),input_extend]);
        filename_in = [input_filename,num2str(ff),input_extend];
        load([path(m,:),filename_in]);
        temp = zeros(size(ObjRecon));
        for i=1:d1
            for j=1:d2
                for k=1:d3
                    if L_temp2(i,j,k)==0
                        temp(i,j,k) = 0;
                    else
                        % % %                         temp(i,j,k) = CalTrace(L_temp2(i,j,k),t);
                        temp(i,j,k) = CalTrace(L_temp2(i,j,k),g(m)+t);
                    end
                end
            end
        end
        Coherence = Coherence + (ObjRecon-Fmean).*temp;
        disp Coherence
        disp(m)
        disp(ff)
        toc;
    end
end
Coherence = Coherence./SD;
Coherence = Coherence/T;
clear ObjRecon;

save([file_path1,'Coherence.mat'],'Coherence');

%% histogram
figure('Name','histogram of Coherence');
histogram(Coherence,'Normalization','pdf');

%% display
index_slice = 1:d3;
num_slices = length(index_slice);
cmap = parula;
pause('on');

max_intensity = max(Coherence(:));

clear slices;
slices(:,:,1:num_slices) = Coherence(:,:,index_slice);
clear picture;
clear RGB;
for l=1:num_slices
    temp = squeeze(L_temp2(:,:,l));
    temp(temp>0)=1;
    temp = double(temp);
    picture(:,:,l) = uint8(slices(:,:,l)/max_intensity*63);
    RGB(:,:,:,l) = ind2rgb(squeeze(picture(:,:,l)),cmap);
    RGB(:,:,1,l) = RGB(:,:,1,l).*temp;
    RGB(:,:,2,l) = RGB(:,:,2,l).*temp;
    RGB(:,:,3,l) = RGB(:,:,3,l).*temp;
    % figure(l);hold off;imshow(picture);hold on;
end

imwrite(squeeze(RGB(:,:,:,1)),[file_path1,'Coherence.tif']);
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),[file_path1,'Coherence.tif'],'WriteMode','append');
end
clear slices;
clear picture;
clear RGB;

%% create a mask according to Coherence
thresh_Coherence = 0.5;
% min_size = 27;
Mask_Coherence = uint16(zeros(size(Coherence)));
Mask_Coherence(Coherence>thresh_Coherence) = 1;

L_temp3 = L_temp2.*Mask_Coherence;
% L_temp3 = bwareaopen(L_temp3,min_size,6);
L_temp3 = imopen(L_temp3,SE); % To exclude small or thin areas and spines, bridges, etc.
L_temp3 = uint16(bwlabeln(L_temp3,6));
num_components_keep3 = max(L_temp3(:));
A3 = sparse(d1*d2*d3,1);
for k=1:num_components_keep3
    temp  = (L_temp3==k);
    A3(:,k) = sparse(reshape(double(temp),[d1*d2*d3,1]));
end
clear Mask_Coherence;

%% display
index_slice = 1:d3;
num_slices = length(index_slice);
temp = uint16(L_temp3);
imwrite(squeeze(temp(:,:,1)),[file_path1,'L_temp3.tif']);
for l=2:num_slices
    imwrite(squeeze(temp(:,:,l)),[file_path1,'L_temp3.tif'],'WriteMode','append');
end

%% extract CalTrace again
K = size(A3,2);
e = size(A3,1);
% % % T = frame_end - frame_start + 1;
T = sum(d);
batch_size = 1;
CalTrace3 = zeros(K,T);
for m=1:b
    for ff=1:batch_size:d(m)
        tic;
        % % % for ff=frame_start:batch_size:frame_end
        Y_r = zeros(e,min([d(m),ff+batch_size-1])-ff+1); % load a batch
        for f=ff:min([d(m),ff+batch_size-1])
        fff = (f - 1)*stepSize + 1;
% % %             ObjRecon = niftiread([file_path,input_filename,num2str(f),input_extend]);
        filename_in = [input_filename,num2str(fff),input_extend];
        load([path(m,:),filename_in]);
            Y_r(:,f-ff+1) = reshape(ObjRecon,[e,1]);
        end
        for k=1:K
            temp = Y_r(A3(:,k)>0,:);
% % %             CalTrace(k,ff:min([d(m),ff+batch_size-1])) = mean(temp,1);
            CalTrace3(k,g(m)+ff:g(m)+min([d(m),ff+batch_size-1])) = mean(temp,1);
        end
        disp CalciumTrace3
        disp(m)
        disp(fff)
        toc;
    end
end
clear Y_r;
CalTrace3_original = CalTrace3;

%% calculate coherence again
CalTrace3 = CalTrace3 - mean(CalTrace3,2);
SD_CalTrace3 = sqrt(sum(CalTrace3.*CalTrace3,2)/T);
SD_CalTrace3 = repmat(SD_CalTrace3, 1,T);
CalTrace3 = CalTrace3./SD_CalTrace3;

Coherence3 = zeros(size(ObjRecon));
for m=1:b
    for t=1:t_step:d(m)
        tic;
        ff = (t - 1)*stepSize + 1;
        % % %     ObjRecon = niftiread([file_path,input_filename,num2str(t),input_extend]);
        filename_in = [input_filename,num2str(ff),input_extend];
        load([path(m,:),filename_in]);
        temp = zeros(size(ObjRecon));
        for i=1:d1
            for j=1:d2
                for k=1:d3
                    if L_temp3(i,j,k)==0
                        temp(i,j,k) = 0;
                    else
                        % % %                         temp(i,j,k) = CalTrace(L_temp2(i,j,k),t);
                        temp(i,j,k) = CalTrace3(L_temp3(i,j,k),g(m)+t);
                    end
                end
            end
        end
        Coherence3 = Coherence3 + (ObjRecon-Fmean).*temp;
        disp Coherence3
        disp(m)
        disp(ff)
        toc;
    end
end
Coherence3 = Coherence3./SD;
Coherence3 = Coherence3/T;
clear ObjRecon;
clear CalTrace3;

save([file_path1,'Coherence3.mat'],'Coherence3','A3','L_temp3','CalTrace3_original');

%% histogram
figure('Name','histogram of Coherence3');
histogram(Coherence3,'Normalization','pdf');

%% display
index_slice = 1:d3;
num_slices = length(index_slice);
cmap = parula;
pause('on');

max_intensity = max(Coherence3(:));

clear slices;
slices(:,:,1:num_slices) = Coherence3(:,:,index_slice);
clear picture;
clear RGB;
for l=1:num_slices
    temp = squeeze(L_temp3(:,:,l));
    temp(temp>0)=1;
    temp = double(temp);
    picture(:,:,l) = uint8(slices(:,:,l)/max_intensity*63);
    RGB(:,:,:,l) = ind2rgb(squeeze(picture(:,:,l)),cmap);
    RGB(:,:,1,l) = RGB(:,:,1,l).*temp;
    RGB(:,:,2,l) = RGB(:,:,2,l).*temp;
    RGB(:,:,3,l) = RGB(:,:,3,l).*temp;
    % figure(l);hold off;imshow(picture);hold on;
end

imwrite(squeeze(RGB(:,:,:,1)),[file_path1,'Coherence3.tif']);
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),[file_path1,'Coherence3.tif'],'WriteMode','append');
end

clear slices;
clear picture;
clear RGB;
clear temp;