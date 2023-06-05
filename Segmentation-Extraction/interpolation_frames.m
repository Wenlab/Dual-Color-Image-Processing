function interpolation_frames(file_path,index_pre,index_post)
	% Syntax: interpolation_frames(file_path,index_pre,index_post)
	%         file_path: the path of data
	%         index_pre: the frame before the first bad frame
	%         index_post: the frame after the last bad frame
	
	% Long description
	%   After alignment and before segmentation, we should pick out the bad frames(motion blur, 
	%	3D reconstruction mistakes, bad alignment, etc.), and estimate these frames by interpolation.

	code_path = cd;
	% file_path = 'G:\\calcium_image\\working\\200703\\12\\';
	cd(file_path);
	load([file_path,'affine_noEyes',num2str(index_pre),'.mat']);
	ObjRecon_pre = ObjRecon;
	load([file_path,'affine_noEyes',num2str(index_post),'.mat']);
	ObjRecon_post = ObjRecon;
	for i=index_pre+1:index_post-1
		a = (i - index_pre)/(index_post - index_pre); % the weight
		ObjRecon = (1-a)*ObjRecon_pre + a*ObjRecon_post;
		eval(['!rename',',affine_noEyes',num2str(i),'.mat',',Ori_affine_noEyes',num2str(i),'.mat']);
		save([file_path,'affine_noEyes',num2str(i),'.mat'],'ObjRecon'); % The original data should be renamed as 'Ori_affine_noEyesi.mat', 
																		% since we will process all files with prefix 'affine_noEyes' in CorrMap_multi.m.

		filename_out = ['affine_noEyes',num2str(i),'.mat'];
		load([file_path,filename_out]);
		MIPs=[max(ObjRecon,[],3) squeeze(max(ObjRecon,[],2));squeeze(max(ObjRecon,[],1))' zeros(size(ObjRecon,3),size(ObjRecon,3))];
		% figure;imagesc(MIPs);axis image;
		MIP=uint16(MIPs);
		eval(['imwrite(MIP,''',file_path,'MIP_affine_noEyes',num2str(i),'_interpolation','.tif'');']);
	end
	cd(code_path);

end