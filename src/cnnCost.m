function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,filterDim,numFilters,poolDim,pred)

%Convert already existing pred variable to false for recalculation 
if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); 
numImages = size(images,3);

% Wc Weight parameter matrix
% bc bias

[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,poolDim,numClasses);

% Gradient of gradient of Weights and Biases.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));


%% Convolutional Layer
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
convDim = imageDim-filterDim+1; 
outputDim = (convDim)/poolDim; 

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

activations = convolveNN(filterDim, numFilters, images, Wc, bc);
activationsPooled = poolNN(poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
activationsPooled = reshape(activationsPooled,[],numImages);

probs = zeros(numClasses,numImages);
M = size(images, 3);

aux1 = Wd*activationsPooled + repmat(bd,1,M);
aux2 = bsxfun(@minus, aux1, max(aux1, [], 1));
aux3 = exp(aux2);
probs = bsxfun(@rdivide, aux3, sum(aux3)); 

clear aux2;
clear aux3;

%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% Calculate Cost
groundTruth = full(sparse(double(labels), 1:M, 1));
aux4 = groundTruth.*probs;
%Extract non-zero entries.
aux5 = log(aux4(aux4 ~= 0)); 
cost = -mean(aux5);
clear aux4;
clear aux5;

% Makes predictions.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
%     preds = probs
    return;
end;

%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% Backpropagation

deriv_1 = (-1/M).*(groundTruth - probs);
clear groundTruth;
Wd_grad = deriv_1*activationsPooled';
clear activationsPooled;
bd_grad = deriv_1*ones(M,1);
deriv_2_pooled_sh = Wd'*deriv_1;
clear deriv_1;
deriv_2_pooled = reshape(deriv_2_pooled_sh,outputDim,outputDim,numFilters,numImages);
deriv_2_upsampled = zeros(convDim,convDim,numFilters,numImages);
for imageNum = 1:numImages
  im = squeeze(images(:,:,imageNum));
  for filterNum = 1:numFilters
    aux3 = (1/(poolDim^2)).*kron(squeeze(deriv_2_pooled(:,:,filterNum,imageNum)),ones(poolDim));
    deriv_2_upsampled(:,:,filterNum,imageNum) = aux3.*activations(:,:,filterNum,imageNum).*(1-activations(:,:,filterNum,imageNum));
    f_now = squeeze(deriv_2_upsampled(:,:,filterNum,imageNum));
    noww = conv2(im,rot90(squeeze(f_now),2),'valid');
    Wc_grad(:,:,filterNum) = squeeze(Wc_grad(:,:,filterNum)) + noww; 
    bc_grad(filterNum) = bc_grad(filterNum) + sum(f_now(:));
    
  end
end

%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%  Gradient Calculation

%% Free Memory
clear activations;
clear deriv_2_pooled;
clear deriv_2_upsampled;

%% Return Gradient for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];
end
