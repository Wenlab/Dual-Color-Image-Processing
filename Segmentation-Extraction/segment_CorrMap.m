%% segment by CorrMap
%% parameters
clear;
% thresh_Var = 0.1;
thresh_Fmax = 9;
thresh_Fmin = 2;
min_size = 27;
adjacent_distance = 3;
adjacent_distance2 = 2; % sqrt(adjacent_distance^2/3)
file_path = 'G:\\calcium_image\\working\\200108_fish1_g6s_7dpf\\';
input_filename = 'affine';
input_extend = '.mat';
frame_start = 1;
frame_end = 2371;
filename_in = [input_filename,num2str(frame_start),input_extend];
% ObjRecon = niftiread([file_path,filename_in]);
load([file_path,filename_in]);
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
Y_mean = Y_mean + ObjRecon;
for f=frame_start+1:frame_end
    filename_in = [input_filename,num2str(f),input_extend];
    % ObjRecon = niftiread([file_path,filename_in]);
    load([file_path,filename_in]);
    Y_mean = Y_mean + ObjRecon;
    clear temp;
    temp(:,:,:,1) = ObjRecon;
    temp(:,:,:,2) = F_max;
    F_max = squeeze(max(temp,[],4));
    temp(:,:,:,2) = F_min;
    F_min = squeeze(min(temp,[],4));
end
Y_mean = Y_mean/(frame_end-frame_start+1);
Y_mean(F_max<thresh_Fmax) = 0; % mask by thresh_Fmax
Y_mean(F_min<thresh_Fmin) = 0; % mask by thresh_Fmin

%% second round: calculate SD,
for f=frame_start:frame_end
    filename_in = [input_filename,num2str(f),input_extend];
    % ObjRecon = niftiread([file_path,filename_in]);
    load([file_path,filename_in]);
    Y_shift = bsxfun(@minus,ObjRecon,Y_mean);
    SD = SD + Y_shift.*Y_shift;
end
SD = sqrt(SD/(frame_end-frame_start+1));
SD(F_max<thresh_Fmax) = 0; % mask by thresh_Fmax
SD(F_min<thresh_Fmin) = 0; % mask by thresh_Fmin

%% third round: calculate Corr,
for f=frame_start:frame_end
    % tic;
    filename_in = [input_filename,num2str(f),input_extend];
    % ObjRecon = niftiread([file_path,filename_in]);
    load([file_path,filename_in]);
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
    % toc
end
% Cov = Cov/(frame_end-frame_start+1);
Corr = Corr/(frame_end-frame_start+1)/14;
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
L_temp2 = bwareaopen(L_temp2,min_size,6);
L_temp2 = uint16(bwlabeln(L_temp2,6));
num_components_keep = max(L_temp2(:));
A = sparse(dx*dy*dz,1);
for k=1:num_components_keep
    temp  = (L_temp2==k);
    A(:,k) = sparse(reshape(double(temp),[dx*dy*dz,1]));
end

%% save A and L_temp2
Fmean = Y_mean;
save([file_path,'CorrMap.mat'],'A','L_temp2','L','SD','F_max','F_min','Fmean','Corr_original');

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

imwrite(squeeze(RGB(:,:,:,1)),[file_path,'CorrMap.tif']);
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),[file_path,'CorrMap.tif'],'WriteMode','append');
end
clear slices;
clear picture;
clear RGB;

temp = uint16(L);
imwrite(squeeze(temp(:,:,1)),[file_path,'L.tif']);
for l=2:num_slices
    imwrite(squeeze(temp(:,:,l)),[file_path,'L.tif'],'WriteMode','append');
end

temp = uint16(L_temp2);
imwrite(squeeze(temp(:,:,1)),[file_path,'L_temp2.tif']);
for l=2:num_slices
    imwrite(squeeze(temp(:,:,l)),[file_path,'L_temp2.tif'],'WriteMode','append');
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
d = size(A,1);
T = frame_end - frame_start + 1;
batch_size = 1;

%%
CalTrace = zeros(K,T);
for ff=frame_start:batch_size:frame_end
    Y_r = zeros(d,min([frame_end,ff+batch_size-1])-ff+1); % load a batch
    for f=ff:min([frame_end,ff+batch_size-1])
        % ObjRecon = niftiread([file_path,input_filename,num2str(f),input_extend]);
        load([file_path,input_filename,num2str(f),input_extend]);
        Y_r(:,f-ff+1) = reshape(ObjRecon,[d,1]);
    end
    for k=1:K
        temp = Y_r(A(:,k)>0,:);
        CalTrace(k,ff:min([frame_end,ff+batch_size-1])) = mean(temp,1);
    end
end

toc

save([file_path,'CalTrace.mat'],'CalTrace');

clear temp;
clear Y_r;

%% calculate the coherence map
%% input: CalTrace, L_temp2, Fmean, SD
%% set parameters
d1 = dx;
d2 = dy;
d3 = dz;

%%
T = frame_end - frame_start + 1;
t_step = 1;
% ObjRecon = niftiread([file_path,input_filename,num2str(1),input_extend]);
load([file_path,input_filename,num2str(1),input_extend]);

%% normalize CalTrace
CalTrace = CalTrace - mean(CalTrace,2);
SD_CalTrace = sqrt(sum(CalTrace.*CalTrace,2)/T);
SD_CalTrace = repmat(SD_CalTrace, 1,T);
CalTrace = CalTrace./SD_CalTrace;

Coherence = zeros(size(ObjRecon));
for t=1:t_step:T
    % ObjRecon = niftiread([file_path,input_filename,num2str(t),input_extend]);
    load([file_path,input_filename,num2str(t),input_extend]);
    temp = zeros(size(ObjRecon));
    for i=1:d1
        for j=1:d2
            for k=1:d3
                if L_temp2(i,j,k)==0
                    temp(i,j,k) = 0;
                else
                    temp(i,j,k) = CalTrace(L_temp2(i,j,k),t);
                end
            end
        end
    end
    Coherence = Coherence + (ObjRecon-Fmean).*temp;
end
Coherence = Coherence./SD;
Coherence = Coherence/T;
clear ObjRecon;

save([file_path,'Coherence.mat'],'Coherence');

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

imwrite(squeeze(RGB(:,:,:,1)),[file_path,'Coherence.tif']);
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),[file_path,'Coherence.tif'],'WriteMode','append');
end
clear slices;
clear picture;
clear RGB;

%% create a mask according to Coherence
thresh_Coherence = 0.5;
min_size = 27;
Mask_Coherence = uint16(zeros(size(Coherence)));
Mask_Coherence(Coherence>thresh_Coherence) = 1;

L_temp3 = L_temp2.*Mask_Coherence;
L_temp3 = bwareaopen(L_temp3,min_size,6);
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
imwrite(squeeze(temp(:,:,1)),[file_path,'L_temp3.tif']);
for l=2:num_slices
    imwrite(squeeze(temp(:,:,l)),[file_path,'L_temp3.tif'],'WriteMode','append');
end

%% extract CalTrace again
K = size(A3,2);
d = size(A3,1);
T = frame_end - frame_start + 1;
batch_size = 1;
CalTrace3 = zeros(K,T);
for ff=frame_start:batch_size:frame_end
    Y_r = zeros(d,min([frame_end,ff+batch_size-1])-ff+1); % load a batch
    for f=ff:min([frame_end,ff+batch_size-1])
        % ObjRecon = niftiread([file_path,input_filename,num2str(f),input_extend]);
        load([file_path,input_filename,num2str(f),input_extend]);
        Y_r(:,f-ff+1) = reshape(ObjRecon,[d,1]);
    end
    for k=1:K
        temp = Y_r(A3(:,k)>0,:);
        CalTrace3(k,ff:min([frame_end,ff+batch_size-1])) = mean(temp,1);
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
for t=1:t_step:T
    % ObjRecon = niftiread([file_path,input_filename,num2str(t),input_extend]);
    load([file_path,input_filename,num2str(t),input_extend]);
    temp = zeros(size(ObjRecon));
    for i=1:d1
        for j=1:d2
            for k=1:d3
                if L_temp3(i,j,k)==0
                    temp(i,j,k) = 0;
                else
                    temp(i,j,k) = CalTrace3(L_temp3(i,j,k),t);
                end
            end
        end
    end
    Coherence3 = Coherence3 + (ObjRecon-Fmean).*temp;
end
Coherence3 = Coherence3./SD;
Coherence3 = Coherence3/T;
clear ObjRecon;
clear CalTrace3;

save([file_path,'Coherence3.mat'],'Coherence3','A3','L_temp3','CalTrace3_original');

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

imwrite(squeeze(RGB(:,:,:,1)),[file_path,'Coherence3.tif']);
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),[file_path,'Coherence3.tif'],'WriteMode','append');
end
clear slices;
clear picture;
clear RGB;

clear temp;