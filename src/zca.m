function [Y X avg] = zca(matFileName, epsilon)
% ZCA: zero-phase whitening transform
    
    cifarTrainData = load(matFileName);
    Z = cifarTrainData.data;
    labels = cifarTrainData.labels;
    
    for i=1:10000
      gray = rgb2gray(reshape(Z(i,:),32,32, 3));
      X(i,:) = reshape(gray', 1, 1024);
    end
  
    X = double(X);
  avg = mean(X, 2);
  X = X - repmat(avg, 1, size(X, 2));

  if nargin == 1
    epsilon = 1e-6;
  end
  
  [N D] = size(X);
  
  % Compute the regularized scatter matrix.
  scatter = (X'*X + epsilon*eye(D));
  
  % The epsilon corresponds to virtual data.
  N = N + epsilon;
  
  % Take the eigendecomposition of the scatter matrix.
  [V D] = eig(scatter);
  
  % This is pretty hacky, but we don't want to divide by tiny
  % eigenvalues, so make sure they're all of reasonable size.
  D = max(diag(D), epsilon);
  
  % Now use the eigenvalues to find the root-inverse of the
  % scatter matrix.
  irD = diag(1./sqrt(D));
  
  % Reassemble into the transformation matrix.
  W = sqrt(N-1) * V * irD * V';
  
  % Apply to the data.
  Y = X*W;
  Y = Y + repmat(avg, 1, size(Y, 2));
  
  for i=1:10000
     images(:,:,i) = reshape(Y(i,:), 32, 32);
  end
  
  save cifarTestGrayImages images labels
  
end