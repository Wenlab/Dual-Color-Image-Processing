%% Segment brain regions using Correlation Map method and extract the calcium trace.

% set path.
file_path_green = 'data/G/regist_green/green_demons/';
file_path_red = 'data/R/regist_red/red_demons/';

% set the prefix name and value name of images.
pre_name_green = 'demons_green_3_';
value_name_green = 'green_demons';
pre_name_red = 'demons_red_3_';
value_name_red = 'red_demons';

% set the distance between two adjacent brain voxels for calculating correlation.
ad_dist = 3;

% set the max and min intensity threshold of dataset.
thresh.max = 9;
thresh.min = 2;

% set the minimum size of segmented regions. 
min_size = 27;

% set the start and end.
start_frame = 8110;
end_frame = 8120;

% extract the green trace.
[Cal_G,Coherence_G,seg_regions,water_corMap_filter,info_data] = corMap(file_path_green,pre_name_green,value_name_green,start_frame,end_frame,ad_dist,thresh,min_size);
disp('Green trace done.');

% extract the red trace.
batch_size = 1;
write_flag = 1;
[Cal_R,Coherence_R] = traceExtract(file_path_red,pre_name_red,value_name_red,seg_regions,water_corMap_filter,info_data,start_frame,batch_size,end_frame,write_flag);
disp('Red trace done.');
