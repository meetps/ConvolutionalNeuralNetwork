function convolvedFeatures = convolveNN(filterDim, numFilters, images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with the given images

numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

%   Convolve every filter with every image here to produce the 

filter = zeros(filterDim,filterDim);
for imageNum = 1:numImages
  for filterNum = 1:numFilters
    convolvedImage = zeros(convDim, convDim);
    filter = squeeze(W(:,:,filterNum));
    filter = rot90(squeeze(filter),2); 
    % Obtain the image
    im = squeeze(images(:, :, imageNum));
    convolvedImage = convolvedImage + conv2(im,filter,'valid');
    % Add the bias unit
    convolvedImage = convolvedImage + b(filterNum);
    convolvedImage = 1./(1+exp(-convolvedImage));
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end


end

