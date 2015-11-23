%Configuration for CNN initialization 

imageDim = 64;
numClasses = 4;
filterDim = 9;    
numFilters = 20;   
poolDim = 2;


%% STL-10 Data Load :
    trainSubset = load('stlTrainSubset.mat');
    subsetImages = trainSubset.trainImages;
    labels = trainSubset.trainLabels;
    images = zeros(64,64,2000);

    % Apply RGB to monoccolorspace transpace i.e (hsv,YCbCr,lab) 
    for k=1:2000
        hsvImage = rgb2hsv(subsetImages(:,:,:,k));
        images(:,:,k) = hsvImage(:,:,3);
    end


%% CIFAR-10 Data Load :
    % for m=1:5
    %     cifarTrainData = load(strcat(['data_batch_' num2str(m) '.mat']));
    %     TrainData = cifarTrainData.data;
    %     TrainLabels = cifarTrainData.labels;
    %     base = 10000*(m-1);
    %     for l=1:10000
    %         rbgimage = reshape(TrainData(l,:),32,32,3);
    %         processedImages = rgb2gray(rbgimage);
    %         images(:,:,base+l) = processedImages(:,:,1);
    %         labels(base+l) = TrainLabels(l)+1;
    %     end
    % end

%% Load CIFAR-10 Whitened Images :  
    % for m=1:5
        % cifarTrainData = load(strcat(['cifarTrainGrayImages' num2str(m) '.mat']));
        % images(:,:,10000*(m-1)+1:10000*(m)) = cifarTrainData.images;
        % labels(10000*(m-1)+1:10000*(m)) = 1+cifarTrainData.labels;
    % end

%% MNIST Data Load
    % images = loadMNISTImages('train-images-idx3-ubyte');
    % images = reshape(images,imageDim,imageDim,[]);
    % images = images(:,:,1:10000);
    % labels = loadMNISTLabels('train-labels-idx1-ubyte');
    % labels(labels==0) = 10; % Remap 0 to 10
    % labels = labels(1:10000,1);

% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);

%%Supervised Learning Parameters
options.epochs = 3;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

%% Calculate the Network weights and Biases using Stochastic Gradient Descent
opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,numFilters,poolDim),theta,images,labels,options);

%%Testing 

%%STL-10 Test Data Load 
    testSubset = load('stlTestSubset.mat');
    testSubsetImages = testSubset.testImages;
    testLabels = testSubset.testLabels;
    testImages = zeros(64,64,3200);

    % Apply RBG to monocolor Transformation
    % HSV Images 
    for k=1:3200
        hsvTestImage = rgb2hsv(testSubsetImages(:,:,:,k));
        testImages(:,:,k) =hsvTestImage(:,:,3); 
    end




%% ZCA Whitened CIFAR-10 Test Data Load
% cifarTestData = load('cifarTestGrayImages.mat');
% testImages = cifarTestData.images;
% testLabels = 1+cifarTestData.labels;

%% CIFAR-10 Test Data Load
    % cifarTestData = load('test_batch.mat');
    % inputImages = cifarTestData.data;
    % testLabels = cifarTestData.labels + 1 ;
    % testImages = zeros(32,32,10000);
 
    %% Apply Color to Monocolorspace Transformation
    % for l=1:10000
    %     rbgimage = reshape(inputImages(l,:),32,32,3);
    %     processedImages = rgb2gray(rbgimage);
    %     testImages(:,:,l) = processedImages(:,:,1);
    % end

    % save cifarTestGray testImages testLabels


%% MNIST Test Data Load
% testImages = loadMNISTImages('t10k-images-idx3-ubyte');
% testImages = reshape(testImages,imageDim,imageDim,[]);
% testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
% testLabels(testLabels==0) = 10; % Remap 0 to 10

%% Retrieve Predicted Values from the Constructed Network using cost funtion
[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
    filterDim,numFilters,poolDim,true);

acc = (sum(preds==testLabels)/length(preds))*100;
    
fprintf('Accuracy is %f percent\n',acc);
