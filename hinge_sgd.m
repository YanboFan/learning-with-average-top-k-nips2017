
function [w,lambda] = hinge_sgd(TrainFea, TrainGnd, ATK_options, SGD_options)

% Learning with Average Top-K Loss: for binary classification, individual
% loss is hinge loss, linear classifier, i.e., y = w^T*x
%
% Input
%   TrainGnd (n*1): 
%           : observed label information, n is the number of samples
%
%   TrainFea (n*d):
%           : observed feature information, d is the number of features        
%
%   ATK_options:
%           : structure holding the hyper-parameters of ATK learning 
%           : ATK_options.K is the number of Top-K losses, 
%           : ATK_options.c is the trade-off parameter
%
%   SGD_options:
%           : strucutre holding the settings of SGD algorithms
%           : SGD_options.NumPass is the number of epochs
%           : SGD_options.stepSize is the initialize stepSize
%
% Output:
%   w (n*1):
%           : learned linear classifier
%   lambda (scalar):
%           : learned lambda in ATK learning, optional
%
% References:
%   YanboFan, Siwei Lyu, Yiming Ying and Bao-Gang Hu. "Learning with
%   Average Top-K Loss", NIPS, 2017.
%
%   version 1.0 --OCT/2017 
%   Written by Yanbo Fan (fanyanbo0124 AT gmail.com)
%

[Ntrain,~] = size(TrainFea); 

% randomly initialize w
rand('seed',3); w = rand(size(TrainFea,2),1);

%initialize lambda
loss = max(0, 1-TrainGnd .* (TrainFea * w));
temp1 = sort(loss,'descend');

if ATK_options.K < Ntrain
    lambda = temp1(ATK_options.K + 1);
else
    lambda = eps;
end
max_lambda = 1; lambda = min(lambda, max_lambda);

%% optimization
NumPass = SGD_options.NumPass;    % epoch
a = SGD_options.stepSize;         % initial stepsize
ATK_c = ATK_options.c;

%
i_pass = 1; t = 1;
avg_w = w; avg_lambda = lambda;

tempData = TrainFea .* repmat(-TrainGnd, 1, size(TrainFea,2)); %n*d
rho_lambda = ATK_options.K / Ntrain; %

%
while i_pass <= NumPass
    for i = 1 : Ntrain
        
        %preparing
        step = a / sqrt(t);
        loss_i = 1 + tempData(i,:)*w; %

        if loss_i > lambda
            w = w - step*(w + ATK_c*tempData(i,:)');
            lambda = lambda - step*ATK_c*(rho_lambda - 1);
        else
            w = w - step*w; lambda = lambda - step*ATK_c*rho_lambda;
        end

        lambda = min(lambda,max_lambda);
        lambda = max(lambda,0);
        
        %save and average
        avg_w = (avg_w*t + w)/(t+1); 
        avg_lambda = (avg_lambda*t + lambda)/(t+1);
        
        t = t + 1;
    end
    i_pass = i_pass + 1;
end
w = avg_w; lambda = avg_lambda; 
end

function [totalloss] = calculate_objective(tempData, lambda, w, ATK_options, Ntrain)

%calculate loss
loss =  1 -lambda + tempData*w;
loss(loss < 0) = 0; 
totalloss = sum(loss) * ATK_options.c / Ntrain + 0.5 * (w'*w) + ...
    ATK_options.c*ATK_options.K*lambda/Ntrain;

end