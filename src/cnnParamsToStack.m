function [Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,poolDim,numClasses)

% Converts parameters for a single layer convolutional neural network followed by a softmax layer 
% into weight tensors/matrices and corresponding biases

outDim = (imageDim - filterDim + 1)/poolDim;
hiddenSize = outDim^2*numFilters;

%% Reshape theta
indS = 1;
indE = filterDim^2*numFilters;
Wc = reshape(theta(indS:indE),filterDim,filterDim,numFilters);
indS = indE+1;
indE = indE+hiddenSize*numClasses;
Wd = reshape(theta(indS:indE),numClasses,hiddenSize);
indS = indE+1;
indE = indE+numFilters;
bc = theta(indS:indE);
bd = theta(indE+1:end);

end
