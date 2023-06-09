%imds = imageDatastore('./dsRed/', FileExtensions=".nii",ReadFcn=@matRead);
dsTrain = imageDatastore('./ds/',FileExtensions=".mat",ReadFcn=@matRead);
TrainCate = imageDatastore('./mask/',FileExtensions=".mat",ReadFcn=@matRead);
dsVal = imageDatastore('./ds2/',FileExtensions=".mat",ReadFcn=@matRead);
ValCate = imageDatastore('./mask2/',FileExtensions=".mat",ReadFcn=@matRead);
toTrain = combine(dsTrain, TrainCate);
toVal = combine(dsVal, ValCate);
%%
lgraph = unet3dLayers([200 152 104], 2);

%%
options = trainingOptions("adam", ...
	  'InitialLearnRate',1e-4,...
	  'MaxEpochs',50,...
	  'MiniBatchSize',4,...
	  'Shuffle','every-epoch',...
      ValidationData=toVal,...
	  Verbose=true,...
      ExecutionEnvironment='multi-gpu');

[net,info]=trainNetwork(toTrain,lgraph,options);


