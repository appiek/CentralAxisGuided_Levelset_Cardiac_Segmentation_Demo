%==========================================================================
% This code is used to cut the images into many pieces for training and
% testing.
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-03-20
%==========================================================================

addpath ./tools/
addpath ./loaddata/
%% set parameters
original_datapath = './data/trainingdata/';
datapath = './data/';
if ~exist(datapath,'dir')
    mkdir(datapath);
end
num_DB = 1; %number of databases
num_Div = 5; % number of spatial division
trainingfolder = 'training-set'; %fold name of samples 
%----------------images augmentation parameters------------------
para_imgaug.maxnum = 1;
para_imgaug.cropsize = 224; %size of the pathces
para_imgaug.random_fliplr = true;
para_imgaug.random_flipup = true;
para_imgaug.random_dropout = false; % drop rate 0~1
para_imgaug.save_format = '.png';

for k=1:num_DB 
    save_to_dir = [datapath trainingfolder '_' int2str(k) '/'];
    if ~exist(save_to_dir,'dir')
        mkdir(save_to_dir);
    end
%     save_to_dir_img = [datapath testingfolder '_' int2str(k) '/'];
%     if ~exist(save_to_dir_img,'dir')
%         mkdir(save_to_dir_img);
%     end
    files = dir([original_datapath '*_seg.nii.gz']);
    OriginImgNames = unique(arrayfun(@(x) x{1}{1},arrayfun(@(x) regexp(x.name,'_', 'split'),files,'UniformOutput',0),'UniformOutput',0));
    for i=1:length(OriginImgNames)
        %% load image mask
        mask_nifti_struct  = load_untouch_nii_gz([original_datapath OriginImgNames{i} '_seg' '.nii.gz']);
        maskdata = mask_nifti_struct.img;
         %% load image data
        Img_nifti_struct  = load_untouch_nii_gz([original_datapath OriginImgNames{i} '.nii.gz']);
        Imgdata = Img_nifti_struct.img; 
        Imgshape = size(Imgdata);
        for m=1:Imgshape(3)
            if m<=2 ||m>Imgshape(3)-2
                para_imgaug.maxnum = 2;
            else
                para_imgaug.maxnum = 1;
            end
                for j=1:Imgshape(4)
                    mask = maskdata(:,:,m,j);
                    if sum(mask(:))==0
                        continue;
                    end
                    Img = uint8(MinMax_Norm(Imgdata(:,:,m,j)));
                    [H,W] = size(mask);
                    if H<para_imgaug.cropsize || W<para_imgaug.cropsize
                        mask = imresize(mask,[para_imgaug.cropsize para_imgaug.cropsize]);
                        Img = imresize(Img,[para_imgaug.cropsize para_imgaug.cropsize]);
                    end
                    [Marker_mask, Marker_weight]= LandmarkGenerator(mask,1);
                    save_prefix = [OriginImgNames{i}  '_' num2str(m) '_' num2str(j)];
                    ImageDataGenerator_withmask(Img,mask,Marker_mask,Marker_weight,save_prefix,save_to_dir,para_imgaug);
                end
                
        end

   end    

end






