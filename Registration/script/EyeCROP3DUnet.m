%% This script is to demostrate how to train and apply
%  the 3D-Unet for zebrafish EyeCrop procedure.
%  Write by Chen Shen, cshen@ustc.edu.cn

%% Prepare DataSet
dsTrain = imageDatastore('./dsTrain/',FileExtensions=".mat",ReadFcn=@matRead);
TrainCate = imageDatastore('./eyeMaskTrain/',FileExtensions=".mat",ReadFcn=@matRead);
dsVal = imageDatastore('./dsValidation/',FileExtensions=".mat",ReadFcn=@matRead);
ValCate = imageDatastore('./eyeMaskValidation/',FileExtensions=".mat",ReadFcn=@matRead);
toTrain = combine(dsTrain, TrainCate);
toVal = combine(dsVal, ValCate);
%% For better training performance, downsize the network
% create the 3D-Unet framework
lgraph = unet3dLayers([200 152 104], 2);

%% Train the net.
options = trainingOptions("adam", ...
	  'InitialLearnRate',1e-4,...
	  'MaxEpochs',50,...
	  'MiniBatchSize',2,...
	  'Shuffle','every-epoch',...
      ValidationData=toVal,...
	  Verbose=true,...
      ExecutionEnvironment='multi-gpu');

[eyeCropNet,info]=trainNetwork(toTrain,lgraph,options);

% Training Procedure on 4 x RTX3090:
% |======================================================================================================================|
% |  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  |  Base Learning  |
% |         |             |   (hh:mm:ss)   |   Accuracy   |   Accuracy   |     Loss     |     Loss     |      Rate       |
% |======================================================================================================================|
% |       1 |           1 |       00:00:10 |       66.52% |       85.09% |       0.8533 |       0.5303 |      1.0000e-04 |
% |       4 |          50 |       00:06:32 |       97.26% |       89.75% |       0.0797 |       0.4648 |      1.0000e-04 |
% |       7 |         100 |       00:11:59 |       98.72% |       89.63% |       0.0385 |       0.5727 |      1.0000e-04 |
% |      10 |         150 |       00:17:26 |       98.39% |       89.83% |       0.0437 |       0.6073 |      1.0000e-04 |
% |      14 |         200 |       00:21:53 |       99.17% |       89.73% |       0.0251 |       0.6413 |      1.0000e-04 |
% |      17 |         250 |       00:25:20 |       99.22% |       89.72% |       0.0229 |       0.6592 |      1.0000e-04 |
% |      20 |         300 |       00:31:47 |       99.40% |       89.66% |       0.0182 |       0.7023 |      1.0000e-04 |
% |      24 |         350 |       00:36:14 |       99.55% |       89.39% |       0.0147 |       0.7322 |      1.0000e-04 |
% |      27 |         400 |       00:40:41 |       99.35% |       89.49% |       0.0186 |       0.7273 |      1.0000e-04 |
% |      30 |         450 |       00:45:09 |       99.67% |       89.57% |       0.0112 |       0.7431 |      1.0000e-04 |
% |      34 |         500 |       00:50:36 |       99.52% |       89.55% |       0.0142 |       0.7597 |      1.0000e-04 |
% |      37 |         550 |       00:56:03 |       99.67% |       89.29% |       0.0105 |       0.7795 |      1.0000e-04 |
% |      40 |         600 |       01:01:29 |       99.68% |       89.42% |       0.0101 |       0.7806 |      1.0000e-04 |
% |      44 |         650 |       01:05:56 |       99.59% |       89.52% |       0.0123 |       0.7718 |      1.0000e-04 |
% |      47 |         700 |       01:10:24 |       99.78% |       89.57% |       0.0076 |       0.7766 |      1.0000e-04 |
% |      50 |         750 |       01:15:50 |       99.78% |       89.54% |       0.0076 |       0.7998 |      1.0000e-04 |
% |======================================================================================================================|
% 
% Training finished: Max epochs completed.

%% Test the net.

% read a nifti (.nii) file
img = niftiread('./ds/Red_1stAffined_221.nii');
img = uint8(img);
% downsample
img_in = imresize3(img, [200,152,104]);

% apply segmentation
[CropMask,scores] = semanticseg(img_in,eyeCropNet ,'MiniBatchSize',1);
CropMask = uint8(CropMask);
CropMask = CropMask -1;
% apply Mask
img_out = img_in.*CropMask;
% resize to normal
img_out = imresize3(img_out, size(img));
fig1 = im2double(img(:,:,100));
fig2 = im2double(img_out(:,:,100));
% demostrate
montage({fig1,fig2});
