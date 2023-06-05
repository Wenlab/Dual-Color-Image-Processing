%% parameters
clear;
% thresh_Var = 0.1;
thresh_Cov = 0.003;
thresh_mean = 10;
file_path = 'D:\\cygwin64\\home\\USER\\eyes_converge_data\\191118_9\\'; %%%%Replace \\
input_filename = 'affine';
input_extend = '.nii';
frame_start = 1;
frame_end = 83;
filename_in = [input_filename,num2str(frame_start),input_extend];
ObjRecon = niftiread([file_path,filename_in]);  %%%%Set AF9 Region
dx = size(ObjRecon,1);
dy = size(ObjRecon,2);
dz = size(ObjRecon,3);
%%%%ObjRecon_R = ObjRecon(10:100,10:100,5:20); %%%%%

%%
Cov = zeros(dx,dy,dz);
Y_mean = zeros(dx,dy,dz);
F_max = zeros(dx,dy,dz);
DFoF_max = zeros(dx,dy,dz);

%% first round: calculate Y_mean, F_max
Y_mean = Y_mean + ObjRecon;
for f=frame_start+1:frame_end
    filename_in = [input_filename,num2str(f),input_extend];
    ObjRecon = niftiread([file_path,filename_in]);
    Y_mean = Y_mean + ObjRecon;
    clear temp;
    temp(:,:,:,1) = ObjRecon;
    temp(:,:,:,2) = F_max;
    F_max = squeeze(max(temp,[],4));
end
Y_mean = Y_mean/(frame_end-frame_start+1);

%% second round: calculate Cov,DFoF_max
for f=frame_start:frame_end
    filename_in = [input_filename,num2str(f),input_extend];
    ObjRecon = niftiread([file_path,filename_in]);
    Y_shift = bsxfun(@minus,ObjRecon,Y_mean);
    for i=2:dx-1
        for j=2:dy-1
            for k=2:dz-1
                % c1 = Y_shift(i,j,k)*Y_shift(i+1,j,k);
                % c2 = Y_shift(i,j,k)*Y_shift(i,j+1,k);
                % c3 = Y_shift(i,j,k)*Y_shift(i,j,k+1);
                % c4 = Y_shift(i,j,k)*Y_shift(i-1,j,k);
                % c5 = Y_shift(i,j,k)*Y_shift(i,j-1,k);
                % c6 = Y_shift(i,j,k)*Y_shift(i,j,k-1);
                % Cov(i,j,k) = Cov(i,j,k)+c1+c2+c3+c4+c5+c6;
                Cov(i,j,k) = Cov(i,j,k) + (Y_shift(i+1,j,k)+Y_shift(i,j+1,k)+Y_shift(i,j,k+1)+Y_shift(i-1,j,k)+Y_shift(i,j-1,k)+Y_shift(i,j,k-1))*Y_shift(i,j,k);
            end
        end
    end
    DFoF = Y_shift./Y_mean;
    clear temp;
    temp(:,:,:,1) = DFoF;
    temp(:,:,:,2) = DFoF_max;
    DFoF_max = squeeze(max(temp,[],4));
end
Cov = Cov/(frame_end-frame_start+1);

min_Cov = min(Cov(:));
max_Cov = max(Cov(:));
Cov = (Cov-min_Cov)/(max_Cov-min_Cov);

L = watershed(1-Cov);

L_temp = L;
num_components = max(L(:));
num_components_keep = 0;
L_temp(Cov<thresh_Cov) = 0;
L_temp(Y_mean<thresh_mean) = 0;
L_temp2 = L_temp;
A = sparse(dx*dy*dz,1);
for k=1:max(L_temp(:))
    temp  = (L_temp==k);
    if sum(temp(:))>0
        num_components_keep = num_components_keep + 1;
        A(:,num_components_keep) = sparse(reshape(double(temp),[dx*dy*dz,1]));
        L_temp2(temp) = num_components_keep;
    end
end

%% save A and L_temp2
save([file_path,'CovMap.mat'],'A','L_temp2','L','DFoF_max','F_max','Y_mean');