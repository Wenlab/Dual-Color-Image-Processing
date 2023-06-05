function demonRegist(file_path_red,file_path_green,start_frame,step_size,end_frame,num_index,template_path,gpu_index)
%% function summary: muti thread regist the fish image using demons method.

%  input:
%   file_path_red/green --- the nii format image directory path of affine registed red/green images.
%   start_frame, step_size, end_frame --- the number of start frame, step size and end frame.
%   num_index --- the transform between template name number and index.
%   template_path --- the path of template.
%   gpu_index --- the gpu id. For multi-GPUs, use a vector.  

%  write: this function will generate 2 directories under both file_path_red and file_path_green.
%   red/green_demons --- contain demon registed red/green images in mat format.
%   red/green_demons_MIPs --- contain the maximum intensity projections in three directions of the demon registed images.

%   Update on 2022.12.03.
    
%% Run.
    % Initialize the parameters.
    if nargin == 7
        gpu_index = [1 2 3 4];
    end
    thread_num = length(gpu_index);
    delete(gcp('nocreate'));
    parpool('local',thread_num);
    spmd_num = ceil((end_frame-start_frame+1)/step_size/thread_num);
    refer_image = niftiread(template_path);

    % create non-rigid nii directory.
    red_regist_path = fullfile(file_path_red,'red_demons');
    red_regist_MIPs_path = fullfile(file_path_red,'red_demons_MIPs');
    green_regist_path = fullfile(file_path_green,'green_demons');
    green_regist_MIPs_path = fullfile(file_path_green,'green_demons_MIPs');
    if ~exist(green_regist_path,"dir")
        mkdir(green_regist_path);
        mkdir(red_regist_path);
        mkdir(red_regist_MIPs_path);
        mkdir(green_regist_MIPs_path);
    end

    % run non-regid registration.
    tic;
    spmd
        gpuDevice(spmdIndex);
        if start_frame+(spmd_num*spmdIndex-1)*step_size <= end_frame
            registDemons(file_path_red,file_path_green,start_frame+spmd_num*(spmdIndex-1)*step_size,step_size,start_frame+(spmd_num*spmdIndex-1)*step_size,num_index,refer_image);
        else
            registDemons(file_path_red,file_path_green,start_frame+spmd_num*(spmdIndex-1)*step_size,step_size,end_frame,num_index,refer_image);
        end
    end
    toc;

end
