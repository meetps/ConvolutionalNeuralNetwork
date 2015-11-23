function theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses)

% Initialize parameters for a single layer convolutional neural network followed by a softmax layer.

%% Initialize parameters randomly based on layer sizes.
assert(filterDim < imageDim,'filterDim must be less that imageDim');

Wc = 1e-1*randn(filterDim,filterDim,numFilters);

% dimension of convolved image
outDim = imageDim - filterDim + 1; 

% assume outDim is multiple of poolDim
assert(mod(outDim,poolDim)==0,...
       'poolDim must divide imageDim - filterDim + 1');

outDim = outDim/poolDim;
hiddenSize = outDim^2*numFilters;

r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;

bc = zeros(numFilters, 1);
bd = zeros(numClasses, 1);

% Convert Wc and bc into a vector, which can then be used with minFunc. 
theta = [Wc(:) ; Wd(:) ; bc(:) ; bd(:)];

end
