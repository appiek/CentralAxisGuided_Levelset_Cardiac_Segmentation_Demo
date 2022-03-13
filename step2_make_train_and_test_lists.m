%==========================================================================
% This code is used to divide the patches into training samples, validation
% samples, testing samples.
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-08-08
%==========================================================================
clc;
clear;
close all;
%% ===============setp1: set parameters==============
datapath = './data/';
num_DB = 1; %number of databases
patchfolder = 'training-set'; %fold name of samples 
nfolds = 5; %determine how many folds we want to use during cross validation
ntaindata = 1; % ntaindata<=nfolds
%% ===============setp2: creat list===================
%% process
for k=1:num_DB
    save_to_dir = [datapath patchfolder '_' int2str(k) '/'];
    files = dir([save_to_dir  '*_l.png']);%ÐÞ¸Äºó×º % we only use images for which we have a mask available, so we use their filenames to limit the OriginImgNames
    patchnames = unique(arrayfun(@(x) x{1}{1},arrayfun(@(x) regexp(x.name,'_l', 'split'),files,'UniformOutput',0),'UniformOutput',0)); %this creates a list of patient id number
    nimgs = length(patchnames);
    indices = crossvalind('Kfold',nimgs,nfolds);
    for iterfold=1:ntaindata %open all of the file Ids for the training and testing files
     %each fold has 4 files created (as discussed in the tutorial)
        fidtrain = fopen(sprintf(['%s' 'patches' '_train_%d.txt'],save_to_dir,iterfold),'w');
        fidtest= fopen(sprintf(['%s'  'patches' '_valid_%d.txt'],save_to_dir,iterfold),'w');
        id_test = find(indices==iterfold);
        id_train = find(indices~=iterfold);
        fprintf(fidtrain,'%s\n',patchnames{id_train});
        fprintf(fidtest,'%s\n',patchnames{id_test});
        fclose(fidtrain);
        fclose(fidtest);
    end
    
end
    
    
    
   


