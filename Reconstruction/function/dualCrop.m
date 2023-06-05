function dualCrop(red_ObjRecon,green_ObjRecon,file_path_red,file_path_green,num,atlas,crop_size)
%% function summary: Crop the black background and rotate the two ObjRecons. 

%  input:
%   red_ObjRecon --- the ObjRecon of reconstructed red frame.
%   green_ObjRecon --- the ObjRecon of reconstructed green frame.
%   file_path_red --- the directory path of red frames.
%   file_path_green --- the directory path of green frames.
%   num --- the number of current frame name.

%  write: This function will generate four directories under both file_path_red and 
% file_path_green.
%   recon_mat --- contain the reconstructed images in mat format.
%   recon_MIPs --- contain the maximum intensity projections in three directions of the reconstructed images.
%   dual_Crop --- Contain images in nii format which were cropped and rotated.
%   dual_MIPs --- contain the maximum intensity projections in three directions of the images in dual_Crop.

%   update 2022.11.30.

%% Make derectories.
    % red directory.
    red_recon_path = fullfile(file_path_red,'recon_mat');
    red_recon_MIP_path = fullfile(file_path_red,'recon_MIPs');
    red_dual_path = fullfile(file_path_red,'dual_Crop');
    red_dual_MIP_path = fullfile(file_path_red,'dual_MIPs');

    % green directory.
    green_recon_path = fullfile(file_path_green,'recon_mat');
    green_recon_MIP_path = fullfile(file_path_green,'recon_MIPs');
    green_dual_path = fullfile(file_path_green,'dual_Crop');
    green_dual_MIP_path = fullfile(file_path_green,'dual_MIPs');

    % create the new directory.
    if ~exist(red_recon_MIP_path,"dir")
        mkdir(red_recon_path);
        mkdir(red_recon_MIP_path);
        mkdir(red_dual_path);
        mkdir(red_dual_MIP_path);
        
        mkdir(green_recon_path);
        mkdir(green_recon_MIP_path);
        mkdir(green_dual_path);
        mkdir(green_dual_MIP_path); 
    end
    
%% Rotate the image reference the atlas.
    % flip the fish because there is a filp transform in our system. 
    red_ObjRecon = flip(red_ObjRecon,3);
    green_ObjRecon = flip(green_ObjRecon,3);
    
    % write the reconstructed images.
    red_recon_name = fullfile(red_recon_path,['red_recon',num2str(num),'.mat']);
    save(red_recon_name,'red_ObjRecon');
    green_recon_name = fullfile(green_recon_path,['green_recon',num2str(num),'.mat']);
    save(green_recon_name,'green_ObjRecon');

    % write the MIP of reconstructed images.
    red_MIP=[max(red_ObjRecon,[],3) squeeze(max(red_ObjRecon,[],2));squeeze(max(red_ObjRecon,[],1))' zeros(size(red_ObjRecon,3),size(red_ObjRecon,3))];
    red_MIP_name = fullfile(red_recon_MIP_path,['red_MIP',num2str(num),'.tif']);
    imwrite(uint16(red_MIP),red_MIP_name);
    green_MIP=[max(green_ObjRecon,[],3) squeeze(max(green_ObjRecon,[],2));squeeze(max(green_ObjRecon,[],1))' zeros(size(green_ObjRecon,3),size(green_ObjRecon,3))];
    green_MIP_name = fullfile(green_recon_MIP_path,['green_MIP',num2str(num),'.tif']);
    imwrite(uint16(green_MIP),green_MIP_name);

%% First, rotate the fish to vertical in XY plane
    red_BW_ObjRecon = red_ObjRecon > mean(mean(mean(red_ObjRecon,'omitnan')+8,'omitnan'),'omitnan');
    stats = regionprops3(red_BW_ObjRecon, 'Volume','Orientation');
    prop = cell2mat(table2cell(stats));

    % Check to see if there is a fish in the image.
    if isempty(prop)
        prop = [1,1,1,1];
    end
    [max_v, index]=max(prop(:,1));
    if max_v < 10^6
        fprintf('the maximium connection volume is %d.\n',max_v);
        disp('the maximium connection volume is too small, it might not be the fish, need to check in person !!!')
    end

    RotateAngle= prop(index,2:4);
    red_ObjRecon=imrotate(red_ObjRecon,-RotateAngle(1),'bicubic', 'crop');
    green_ObjRecon=imrotate(green_ObjRecon,-RotateAngle(1),'bicubic', 'crop');
    
%% second, check if the fish is right vertival whose head in the top using template matching, if not flip it.
    red_xy_MIP = max(red_ObjRecon,[],3);
    zbb_xy_MIP = max(atlas,[],3);
    cross_corr = normxcorr2(zbb_xy_MIP,red_xy_MIP);

    zbb_xy_MIP_flip = max(imrotate(atlas,180,'bicubic', 'crop'),[],3);
    flip_corr = normxcorr2(zbb_xy_MIP_flip,red_xy_MIP);

    if max(flip_corr,[],'all') > max(cross_corr,[],'all')
        red_ObjRecon = imrotate(red_ObjRecon,180,'bicubic', 'crop');
        green_ObjRecon = imrotate(green_ObjRecon,180,'bicubic', 'crop');
    end

%% third, rotate the fish vertical in XZ plane.
    red_BW_ObjRecon = red_ObjRecon > mean(mean(mean(red_ObjRecon,'omitnan')+8,'omitnan'),'omitnan');
    statsX = regionprops3(red_BW_ObjRecon,'volume', 'Orientation');
    propX = cell2mat(table2cell(statsX));

    if isempty(propX)
        propX = [1,1,1,1];
    end
    [~, index]=max(propX(:,1));
    RotateAngle = propX(index,2:4);
    
    red_ObjRecon=permute(red_ObjRecon,[3 1 2]);
    red_ObjRecon = imrotate(red_ObjRecon,-RotateAngle(2),'bicubic', 'crop');
    red_ObjRecon=permute(red_ObjRecon,[2 3 1]);
    green_ObjRecon=permute(green_ObjRecon,[3 1 2]);
    green_ObjRecon = imrotate(green_ObjRecon,-RotateAngle(2),'bicubic', 'crop');
    green_ObjRecon=permute(green_ObjRecon,[2 3 1]);

%% fourth, crop the background of the initial size (600*600*300 for us) to cropped size (308*400*210 for us).
    red_BW_ObjRecon = red_ObjRecon > mean(mean(mean(red_ObjRecon,'omitnan')+8,'omitnan'),'omitnan');
    statsX = regionprops3(red_BW_ObjRecon,'volume','Centroid');
    propX = cell2mat(table2cell(statsX));
    
    if isempty(propX)
        propX = [1,1,1,1];
    end
    [~, index]=max(propX(:,1));
    CentroID = propX(index,2:4);

    % intial size.
    intial_size = size(red_ObjRecon);

    % get the first dimension size +- 5 which for interpolation.
    if CentroID(1) < crop_size(1)/2+5
        CentroID(1) = crop_size(1)/2+5;
    else
        if CentroID(1) > intial_size(1)-crop_size(1)/2-5
            CentroID(1) = intial_size(1)-crop_size(1)/2-5;
        end
    end

    % get the second dimension size +- 5 which for interpolation and translation 80 for fine tuning.
    CentroID(2) = CentroID(2) - 80;
    if CentroID(2) < crop_size(2)/2+5
        CentroID(2) = crop_size(2)/2+5;
    end
    if CentroID(2) > intial_size(2)-crop_size(2)/2-5
        CentroID(2) = intial_size(2)-crop_size(2)/2-5;
    end

    % get the boundary of fish.
    if CentroID(3) < crop_size(3)/2+3
        image_size = [round(CentroID(1)-crop_size(1)/2-4),round(CentroID(1)+crop_size(1)/2+5), ...
            round(CentroID(2)-crop_size(2)/2-4),round(CentroID(2)+crop_size(2)/2+5),...
            1,crop_size(3)+6];
        flag = 1;
    else
        if CentroID(3) > intial_size(3)-crop_size(3)/2-3
            image_size = [round(CentroID(1)-crop_size(1)/2-4),round(CentroID(1)+crop_size(1)/2+5), ...
                round(CentroID(2)-crop_size(2)/2-4),round(CentroID(2)+crop_size(2)/2+5),...
                intial_size(3)-crop_size(3)-5,intial_size(3)];
            flag = 2;
        else
            image_size = [round(CentroID(1)-crop_size(1)/2-4),round(CentroID(1)+crop_size(1)/2+5), ...
                round(CentroID(2)-crop_size(2)/2-4),round(CentroID(2)+crop_size(2)/2+5),...
                round(CentroID(3)-crop_size(3)/2-2),round(CentroID(3)+crop_size(3)/2+3)];
            flag = 3;
        end
    end
    
    % Warn: the image correspond X and Y is different.
    red_ObjRecon = red_ObjRecon(image_size(3):image_size(4),image_size(1):image_size(2),image_size(5):image_size(6));
    green_ObjRecon = green_ObjRecon(image_size(3):image_size(4),image_size(1):image_size(2),image_size(5):image_size(6));
    
%% Finaly, interp the image and save them.
    [X_bound,Y_bound,Z_bound] = size(red_ObjRecon);
    % Warn: the image correspond X and Y is different.
    [X,Y,Z] = meshgrid(linspace(1,Y_bound,crop_size(1)+10),linspace(1,X_bound,crop_size(2)+10),linspace(1,Z_bound,crop_size(3)+6));
    
    % interp the image.
    red_interp = uint16(interp3(red_ObjRecon,X,Y,Z,'spline'));
    green_interp = uint16(interp3(green_ObjRecon,X,Y,Z,'spline'));
    
    % write the image to nii file.
    if flag == 1
        red_interp = red_interp(6:crop_size(2)+5,6:crop_size(1)+5,1:crop_size(3));
        green_interp = green_interp(6:crop_size(2)+5,6:crop_size(1)+5,1:crop_size(3));
    else
        if flag == 2
            red_interp = red_interp(6:crop_size(2)+5,6:crop_size(1)+5,7:crop_size(3)+6);
            green_interp = green_interp(6:crop_size(2)+5,6:crop_size(1)+5,7:crop_size(3)+6);
        else
            red_interp = red_interp(6:crop_size(2)+5,6:crop_size(1)+5,4:crop_size(3)+3);
            green_interp = green_interp(6:crop_size(2)+5,6:crop_size(1)+5,4:crop_size(3)+3);
        end
    end

    red_Filename_Out = ['Red',num2str(num),'.nii'];
    green_Filename_Out = ['Green',num2str(num),'.nii'];
    niftiwrite(red_interp,fullfile(red_dual_path,red_Filename_Out));
    niftiwrite(green_interp,fullfile(green_dual_path,green_Filename_Out));
    
    % write the MIP file.
    RescaledRed_Mip = [max(red_interp,[],3) squeeze(max(red_interp,[],2));squeeze(max(red_interp,[],1))' zeros(size(red_interp,3),size(red_interp,3))];
    RescaledRed_Mip = uint16(RescaledRed_Mip);
    imwrite(RescaledRed_Mip,fullfile(red_dual_MIP_path,['MIP_Red','_',num2str(num),'.tif']));
    RescaledGreen_Mip = [max(green_interp,[],3) squeeze(max(green_interp,[],2));squeeze(max(green_interp,[],1))' zeros(size(green_interp,3),size(green_interp,3))];
    RescaledGreen_Mip = uint16(RescaledGreen_Mip);
    imwrite(RescaledGreen_Mip,fullfile(green_dual_MIP_path,['MIP_Green','_',num2str(num),'.tif']));
    
end