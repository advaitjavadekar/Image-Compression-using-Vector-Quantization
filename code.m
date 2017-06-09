image_read=imread('A:\Study material\ECE 592 Data Science\projects\luna.jpg');
image_gray = rgb2gray(image_read);
number_of_rows = size(image_gray,1);
number_of_columns = size(image_gray,2);
m = 1024;%input('specify the number of rows in the resized image: ');
n = 1024;%input('specify the number of columns in the resized image: ');
image = imresize(image_gray, [m,n]);
p = 4;%input('Enter the size of the patch: ');
p_square = p*p;
r= input('Enter the Rate value: ');

ipatch = zeros(p);
numberofpatches = round((m*n)/p_square);

%patch to store pixel values of every patch in every single row
patch = zeros(numberofpatches,p_square);

clusters= round(2^(r*p_square));

k=1;
for i= 1:p:m-p+1
    for j= 1:p:n-p+1
        %partitioning image into patches of required size 
        ipatch = image(i:i+p-1,j:j+p-1);
        %storing pixel values of every patch
        patch(k,:)= reshape(ipatch', 1, p*p);
        k=k+1;
    end
end

%idx stores cluster indices of every observtion in a
%(numberofpatches X 1)matrix

%center stores the centroid locations of each cluster in a
%(clusters X p_square) matrix

[idx, center] = kmeans(patch, clusters, 'MaxIter', 100);

patch_new = patch;


%to round off values at center
for i = 1:numberofpatches
    for j = 1:clusters
        if (idx(i) == j)
              patch_new(i,:) = round(center(j,:));
        end
    end
end

  
image_compressed = image;


%new patch values being stored in ipatch
%transpose of ipatch being saved into image_compressed
k=1;
for i= 1:p:m-p+1
    for j= 1:p:n-p+1
        ipatch = reshape(patch_new(k,:),p,p);
        image_compressed(i:i+p-1,j:j+p-1) = ipatch';
        k=k+1;
    end
end
figure
imshow([image, image_compressed]);
title('Original and Compressed Image');
hold
figure
imshow(image);
title('original image');
hold
partofimage=zeros(100,100);
partofcompimage=zeros(100,100);
for i=1:1:300
    for j=1:1:300
        partofimage(i,j)=(image(i,j));
        partofcompimage(i,j)=(image_compressed(i,j));
    end
end
partofimage = uint8(partofimage);
partofcompimage = uint8(partofcompimage);
figure
imshow([partofimage, partofcompimage]);
title('parts of original and compressed image');

% calculating distortion
squared_error=double(0);
for i=1:m
    for j=1:n
        squared_error = squared_error + (double((image(i,j)-image_compressed(i,j))^2));
    end
end
distortion = squared_error/(m*n);

%matrix ProbabilityofClusters saves probability of each cluster
% better compression
[CountofEachCluster,ClusterNumber]=hist(idx,unique(idx));
ProbabilityofClusters = CountofEachCluster'/numberofpatches;

MaxProb= max(ProbabilityofClusters);
MinProb= min(ProbabilityofClusters);

%average coding length required
entropy_positive = 0;
for i = 1:clusters
entropy_positive = entropy_positive + (ProbabilityofClusters(i)*log2(ProbabilityofClusters(i)));
end
entropy = -(entropy_positive);
average_coding_length=ceil(entropy);
NormalizedRate = entropy/p_square;


%fprintf('entropy =%d \n',entropy);
%fprintf('MaxProb = %d\n',MaxProb);
%fprintf('MinProb = %d\n',MaxProb);
fprintf('distortion =%d \n',distortion);
fprintf('Average Coding Length =%d bits\n',average_coding_length);
fprintf('Normalized Rate =%d \n',NormalizedRate);