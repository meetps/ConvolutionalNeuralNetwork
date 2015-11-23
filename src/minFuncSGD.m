function [opttheta] = minFuncSGD(funObj,theta,data,labels,options)

%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(labels);
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));

%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% Minimizing Loss funtion using Stochastic Gradient Routine
it = 0;
for e = 1:epochs
    rp = randperm(m);    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;
        if it == momIncrease
            mom = options.momentum;
        end;
        mb_data = data(:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));
        [cost grad] = funObj(theta,mb_data,mb_labels);
        velocity = (mom.*velocity) + (alpha.*grad);
        theta = theta - velocity;

        fprintf('Epoch %d: Cost at iter  %d is %f\n',e,it,cost);
    end;
    alpha = alpha/2.0;
end;

opttheta = theta;
end
