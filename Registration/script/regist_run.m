%% Note: You must run the adpath.m script firstly, then run this code under Registeration path.

%% Run demons registration.

% Set environment.
cd ../;
adpath;

% set the red and green directory path of affine registed images.
file_path_red = 'data/R/regist_red';
file_path_green = 'data/G/regist_green';
file_path_template = fullfile(file_path_red,'red_demons');

% extract the name number of affine registed images.
tif_struct = dir(fullfile(file_path_red,'regist_red_1_*.nii'));
all_tifs = {tif_struct.name};
len = size(tif_struct,1);
num_index = zeros(len,1);
for i = 1:len
    name_num = split(all_tifs{i},'.');
    name_num = split(name_num{1},'_');
    name_num = name_num{4};
    num_index(i) = str2double(name_num);
end

% Check the number of bad templates in person and write the name number here.
% Note: it is same as the template_run, so only check it once.
false_num = [];
start_num = 1;
end_num = length(dir(fullfile(file_path_red,'regist_red_1_*.nii')));
mask_path = 'data/Mask/Mask.mat';
thread_num = 28;

% for atlas template.
template_path_1 = 'data/Atlas/Ref-zbb2.nii';
step_size_1 = 1; %(recommend 100)

% for mean template.
step_size_2 = 1;
gpu_index = [1];
mode = 2;

% crop the eyes.
eyesCrop(file_path_red,file_path_green,start_num,step_size_2,end_num,num_index,mask_path,thread_num);

% demons registration for atlas.
demonRegist(file_path_red,file_path_green,start_num,step_size_1,end_num,num_index,template_path_1,gpu_index);

% mean the template.
templateMean(file_path_template,num_index,false_num,mode);

% demons registration for mean template.
template_path_2 = fullfile(file_path_template,'mean_template.nii');
demonRegist(file_path_red,file_path_green,start_num,step_size_2,end_num,num_index,template_path_2,gpu_index);
