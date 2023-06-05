%% calculate the coherence map
%% input: CalTrace, L_temp2, Fmean, SD
%% set parameters
file_path = 'D:\\cygwin64\\home\\USER\\20_01_08_05\\';
input_filename = 'affine';
input_extend = '.nii';
frame_start = 1;
frame_end = 200;
d1 = 600;
d2 = 600;
d3 = 280;

%%
T = frame_end - frame_start + 1;
t_step = 1;
ObjRecon = niftiread([file_path,input_filename,num2str(1),input_extend]);

%% normalize CalTrace
CalTrace = CalTrace - mean(CalTrace,2);
SD_CalTrace = sqrt(sum(CalTrace.*CalTrace,2)/T);
SD_CalTrace = repmat(SD_CalTrace, 1,T);
CalTrace = CalTrace./SD_CalTrace;

Coherence = zeros(size(ObjRecon));
for t=1:t_step:T
	ObjRecon = niftiread([file_path,input_filename,num2str(t),input_extend]);
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

imwrite(squeeze(RGB(:,:,:,1)),'Coherence.tif');
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),'Coherence.tif','WriteMode','append');
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
imwrite(squeeze(temp(:,:,1)),'L_temp3.tif');
for l=2:num_slices
    imwrite(squeeze(temp(:,:,l)),'L_temp3.tif','WriteMode','append');
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
        ObjRecon = niftiread([file_path,input_filename,num2str(f),input_extend]);
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
	ObjRecon = niftiread([file_path,input_filename,num2str(t),input_extend]);
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

imwrite(squeeze(RGB(:,:,:,1)),'Coherence3.tif');
for l=2:num_slices
    imwrite(squeeze(RGB(:,:,:,l)),'Coherence3.tif','WriteMode','append');
end
clear slices;
clear picture;
clear RGB;

clear temp;