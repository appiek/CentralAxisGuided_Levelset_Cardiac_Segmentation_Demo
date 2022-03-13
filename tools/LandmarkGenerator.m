%==========================================================================
% This functiom is used to generate interval between objects for image
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-08-08
%==========================================================================
function [Marker_mask, Marker_weight]= LandmarkGenerator(mask,scale)
[H,W]=size(mask);
%% step1:距离变换
mask = bwmorph(mask ,'thin',Inf);
Marker_mask = bwmorph(mask,'dilate',scale);
D = -bwdist(~mask);
%% step3:计算权值
Num1 = sum(Marker_mask(:));
TotalNum = H*W;
Num2 = TotalNum - Num1;
Weight_class = Marker_mask.*(Num2/TotalNum) + (~Marker_mask).*(Num1/TotalNum)*10;

Marker_weight = double(Weight_class + Marker_mask.*(1./(1+exp(D./10))));
