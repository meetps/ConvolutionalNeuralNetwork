function pooledFeatures = poolNN(poolDim, convolvedFeatures)

%cnnPool Pools the given convolved features

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

%   Now pool the convolved features in regions of poolDim x poolDim, to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, 

q_tot = poolDim^2;
avg_kern = ones(poolDim)/q_tot;

for imageNum = 1:numImages
 	for filterNum = 1:numFilters
		%Convolution of the current image with average kernel:
	    current_image = squeeze(convolvedFeatures(:,:,filterNum,imageNum));
	    Img_conv = conv2(current_image,avg_kern,'valid');
		%Downsampling to get the correct
	    aux = downsample(Img_conv,poolDim);
	    aux1 = downsample(aux',poolDim);
	    aux1 = aux1';  
	    pooledFeatures(:,:,filterNum,imageNum) = aux1;
  	end
end

end

