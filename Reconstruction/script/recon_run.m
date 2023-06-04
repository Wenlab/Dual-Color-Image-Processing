%% Note: You must run the adpath.m script firstly, then run this code under Reconstruction path.

%% A demo to run image reconstruction and crop the black background.

% Directory path of the tif multi-view light field images.
% Note: in the demo, the name of the images is like '00000001.tif'.

file_Path_Red = 'data/R/';
file_Path_Green = 'data/G/';

% Point spread function.
PSF_path_red = 'data/PSF/PSF_R.mat';
PSF_path_green = 'data/PSF/PSF_G.mat';
red_PSF = load(PSF_path_red).PSF_1;
green_PSF = load(PSF_path_green).PSF_1;

% Atlas.
atlas_path = 'data/Atlas/Ref-zbb1.nii';
atlas = niftiread(atlas_path);

% Affine transform between the green and the red.
load('data/Transform/Affine_G2R.mat','tform');

% set the output size which was cropped.
crop_size = [308,400,210];

% Initialize the parameters.
start_num = 1;
step_size = 1;
end_num = length(dir(fullfile(file_Path_Red,'*.tif')));

% Set gpu index (if have multi-gpus, use a vector.).
gpu_index = [1];

% Run reconstruction and crop the black background.
reConstruction(file_Path_Red,file_Path_Green,red_PSF,green_PSF,atlas,crop_size,start_num,step_size,end_num,tform,gpu_index);

