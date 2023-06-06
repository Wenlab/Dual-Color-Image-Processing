%% Note: You must run the adpath.m script firstly, then run this code under Registeration path.

%% Bad image interpolation.

% Set environment.
cd ../;
adpath;

% Set path.
file_path_green = '../data/G/regist_green/green_demons';
file_path_red = '../data/R/regist_red/red_demons';

% Set prefix name of image.
pre_name_red = 'demons_red_3_';
pre_name_green = 'demons_green_3_';

% Check the bad image index in person and write them down here.
bad_index = sort([]);

%% Interp the bad index.
num = 0;
i = 1;
while i <= length(bad_index)
    if i < length(bad_index)
        if bad_index(i) == bad_index(i+1)-1
            num = num +1;
            i = i+1;
            continue;
        end
    end
    disp(['start continuous index: ', num2str(bad_index(i-num))]);
    disp(['end continuous index: ', num2str(bad_index(i))]);
    interp_bad(file_path_red,pre_name_red,bad_index(i-num)-1,bad_index(i)+1,1);
    interp_bad(file_path_green,pre_name_green,bad_index(i-num)-1,bad_index(i)+1,0);
    num = 0;
    i = i+1;
end
