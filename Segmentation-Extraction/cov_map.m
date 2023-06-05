%% parameters
clear;
% thresh_Var = 0.1;
thresh_Cov = 0.003;
thresh_mean = 10;
dx = 600;
dy = 600;
dz = 280;
T = 400;
file_path = 'D:\\cygwin64\\home\\USER\\data019_20mA\\';
load([file_path,'patch_index.mat']); % patch_index, patch_range
num_patches = size(patch_range,1);
input_filename = 'registered_sequence_patch';
input_extend = '.mat';

%%
Cov = zeros(dx,dy,dz);
Y_mean = zeros(dx,dy,dz);
for p=1:num_patches
    clear Y;
    load([file_path,input_filename,num2str(p),input_extend]); % load a patch, Y
    Y_mean_patch = mean(Y,4);
    Y_mean(patch_range(p,1):patch_range(p,2),patch_range(p,3):patch_range(p,4),patch_range(p,5):patch_range(p,6)) = Y_mean_patch;
    Y_shift = bsxfun(@minus,Y,Y_mean_patch);
    for x=patch_range(p,1)+1:patch_range(p,2)-1
        for y=patch_range(p,3)+1:patch_range(p,4)-1
            for z=patch_range(p,5)+1:patch_range(p,6)-1
                i = x - patch_range(p,1) + 1;
                j = y - patch_range(p,3) + 1;
                k = z - patch_range(p,5) + 1;
                c1 = mean(Y_shift(i,j,k,:).*Y_shift(i+1,j,k,:));
                c2 = mean(Y_shift(i,j,k,:).*Y_shift(i,j+1,k,:));
                c3 = mean(Y_shift(i,j,k,:).*Y_shift(i,j,k+1,:));
                c4 = mean(Y_shift(i,j,k,:).*Y_shift(i-1,j,k,:));
                c5 = mean(Y_shift(i,j,k,:).*Y_shift(i,j-1,k,:));
                c6 = mean(Y_shift(i,j,k,:).*Y_shift(i,j,k-1,:));
                Cov(x,y,z) = c1+c2+c3+c4+c5+c6;
            end
        end
    end
end

% load('D:\\cygwin64\\home\\USER\\g6f_8dpf_10fps_100ms_190715_100mA_data021_PurpleDuration1s_300um_150mW488_1_1\\0\\DownSampledData.mat');

% Y_mean = mean(Y,4);
% Y_shift = bsxfun(@minus,Y,Y_mean);

% dx = size(Y,1);
% dy = size(Y,2);
% dz = size(Y,3);
% T = size(Y,4);
% Cov = zeros(size(Y_mean));
% % Corr = zeros(size(Y_mean));
% % Var = zeros(size(Y_mean));
% % flag_Var0 = false(size(Y_mean));

% % Var = mean(Y_shift.^2,4);
% % flag_Var0(Var<thresh_Var) = true;
% % flag_Var_surround0 = flag_Var0;

% for i=2:dx-1
%     for j=2:dy-1
%         for k=2:dz-1
%             c1 = mean(Y_shift(i,j,k,:).*Y_shift(i+1,j,k,:));
%             c2 = mean(Y_shift(i,j,k,:).*Y_shift(i,j+1,k,:));
%             c3 = mean(Y_shift(i,j,k,:).*Y_shift(i,j,k+1,:));
%             c4 = mean(Y_shift(i,j,k,:).*Y_shift(i-1,j,k,:));
%             c5 = mean(Y_shift(i,j,k,:).*Y_shift(i,j-1,k,:));
%             c6 = mean(Y_shift(i,j,k,:).*Y_shift(i,j,k-1,:));
%             Cov(i,j,k) = c1+c2+c3+c4+c5+c6;
%             % Corr(i,j,k) = (c1/sqrt(Var(i+1,j,k)) + c2/sqrt(Var(i,j+1,k)) + c3/sqrt(Var(i,j,k+1)) + c4/sqrt(Var(i-1,j,k)) + c5/sqrt(Var(i,j-1,k)) + c6/sqrt(Var(i,j,k-1)))/sqrt(Var(i,j,k));
%             % if flag_Var0(i,j,k)
%             %     flag_Var_surround0(i+1,j,k) = true;
%             %     flag_Var_surround0(i,j+1,k) = true;
%             %     flag_Var_surround0(i,j,k+1) = true;
%             %     flag_Var_surround0(i-1,j,k) = true;
%             %     flag_Var_surround0(i,j-1,k) = true;
%             %     flag_Var_surround0(i,j,k-1) = true;
%             % end
%         end
%     end
% end

min_Cov = min(Cov(:));
max_Cov = max(Cov(:));
Cov = (Cov-min_Cov)/(max_Cov-min_Cov);

% min_Corr = min(Corr(~flag_Var_surround0));
% max_Corr = max(Corr(~flag_Var_surround0));
% Corr(flag_Var_surround0) = min_Corr;
% Corr = (Corr-min_Corr)/(max_Corr-min_Corr);

L = watershed(1-Cov);

L_temp = L;
num_components = max(L(:));
num_components_keep = 0;
L_temp(Cov<thresh_Cov) = 0;
L_temp(Y_mean<thresh_mean) = 0;
A = sparse(dx*dy*dz,1);
for k=1:max(L_temp(:))
    temp  = (L_temp==k);
    if sum(temp(:))>0
        num_components_keep = num_components_keep + 1;
        A(:,num_components_keep) = sparse(reshape(double(temp),[dx*dy*dz,1]));
    end
end

% num_components = max(L(:));
% num_components_keep = 0;
% foot_print = zeros(dx,dy,dz);
% for k=1:num_components
%     peak_Cov = max(Cov(L==k));
%     if peak_Cov>thresh_Cov
%         num_components_keep = num_components_keep + 1;
%         foot_print_k = false(dx,dy,dz);
%         temp = foot_print_k;
%         foot_print_k(L==k) = true;
%         temp(Cov>thresh_Cov) = true;
%         foot_print_k = foot_print_k & temp;
%         foot_print = foot_print + foot_print_k*num_components_keep;
%     end
% end


% figure;
% imagesc(foot_print(:,:,91));

% figure;
% imagesc(Y_mean(:,:,91));
% figure;
% D = Cov(:,:,91);
% imagesc(D);
% % figure;
% % imagesc(Corr(:,:,91));

% L = watershed(1-D);
% Boundary = L;
% Boundary(L>0) = 1;
% Boundary = double(Boundary);

% RGB = ind2rgb(uint8((D-min(D(:)))/(max(D(:))-min(D(:)))*63),parula);
% RGB(:,:,1) = RGB(:,:,1).*Boundary;
% RGB(:,:,2) = RGB(:,:,2).*Boundary;
% RGB(:,:,3) = RGB(:,:,3).*Boundary;
% figure
% imshow(RGB);

% temp = Y_mean(:,:,91);
% RGB = ind2rgb(uint8((temp-min(temp(:)))/(max(temp(:))-min(temp(:)))*63),parula);
% RGB(:,:,1) = RGB(:,:,1).*Boundary;
% RGB(:,:,2) = RGB(:,:,2).*Boundary;
% RGB(:,:,3) = RGB(:,:,3).*Boundary;
% figure
% imshow(RGB);