%% Note: You must run the adpath.m script firstly, then run this code under Registeration path.

%% Run affine registration.

% Set environment.
cd ../;
adpath;

% set the directory path of template images.
file_path_template = "data/R/template";

% extract the name number of template images.
tif_struct = dir(fullfile(file_path_template,'template*.nii'));
all_tifs = {tif_struct.name};
len = size(tif_struct,1);
num_index = zeros(len,1);
for i = 1:len
    file_name = all_tifs{i};
    name_num = isstrprop(file_name,'digit');
    num_index(i) = str2double(file_name(name_num));
end

% Check the number of bad templates in person and write the name number here.
false_num = [];
mode = 1;

% Run the templateMean.
templateMean(file_path_template,num_index,false_num,mode);