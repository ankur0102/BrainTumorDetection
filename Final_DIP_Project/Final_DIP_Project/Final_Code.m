clear all;
clc;
% Selecting an image form the file path
[I,imgPath]=uigetfile('*.jpg;*.png;*.bmp','select a input image');
imgPath=strcat(imgPath,I);
img1=imread(imgPath);
%Checks If Image is RGB , If it is convert it to gray scale image
if(size(img1,3)>2)
img1=rgb2gray(img1);
end
img1 = imresize(img1,[200,200]);
% Converts uint8 to Double
img=im2double(img1);
%Removing Noise by applying Median Filter
img=medfilt2(img);
%Converting Image into Binary Image with a threshold value 0.65
bw = im2bw(img,0.7);
imgCheck = im2bw(img1);
% Crearing 2D Binary Image Label
label = bwlabel(bw);
% For Finding the Properties of the image
% Returns a scalar specifying the proportion of the pixels in the convex hull that are also in the region
properties = regionprops(label,'Solidity','Area');
% Density of the convex Hull
density = [properties.Solidity];
% Area of the Hull
area = [properties.Area];
% High Density Areas are   tumor
denseCond = density > 0.5;
% Finding the biggest tumor
denseMax = max(area(denseCond));
tumorFinal = find(area == denseMax);
% From the properties finded out earlier, we found the biggest tumor
tumor = ismember(label,tumorFinal);
% Using Morphological Operation using square structural Element
se = strel('square',3);
 tumor = imdilate(tumor,se);

% Displaying the Image
figure(1)
imshow(img,[]);
title('Brain MRI Image');

figure(2)
imshow(tumor,[]);
title('Extracted Tumor');

[boundary,l1] = bwboundaries(tumor,'noholes');
figure(3)

imshow(img,[]);
title('Tumor Boundary');
hold on;
for i = 1:length(boundary)
    plot(boundary{i}(:,2),boundary{i}(:,1),'r','linewidth',1.45); 
end
title('Detected Tumor')
hold off;


s1 = imgCheck(:,:);

[c1,d1,e1,f1] = dwt2(s1,'db4');
[c2,d2,e2,f2] = dwt2(c1,'db4');
[c3,d3,e3,f3] = dwt2(c2,'db4');

dwtfeat = [c3,d3,e3,f3];
% Performing the Principal Component Extraction for feature extraction
G = pca(dwtfeat);

g = graycomatrix(G);
statistics = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = statistics.Contrast;
Correlation = statistics.Correlation;
Energy = statistics.Energy;
Homogeneity = statistics.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
m = size(G,1); % No. of Rows
n = size(G,2); % No. of Columns
diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        diff = diff + temp;
    end
end
IDM = double(diff);   
features = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
% The 30 datasets which we are training the SVM with
load myset.mat % 
 svmStruct = svmtrain(meas,label,'kernel_function', 'linear');
 Species = svmclassify(svmStruct,features,'showplot',false);
 
 if strcmpi(Species,'MALIGNANT')
     disp('Malignant Tumor ');
     
 else
     disp(' Benign Tumor ');
   
 end
