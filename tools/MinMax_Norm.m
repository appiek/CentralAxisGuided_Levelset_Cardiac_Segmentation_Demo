function Img_process = MinMax_Norm(Img)
minval = min(Img(:));
maxval = max(Img(:));
Img_process = 255*(Img-minval)/(maxval -minval);