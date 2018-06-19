
clear; clc;

data_path = 'synthetic_data/data1.mat';
resPath = 'logistic_result/result_data1';

load(data_path);

%preparing data
TrainGnd = y; %n*1
TrainFea = x; %n*d
Ntrain = size(TrainFea,1);
TrainFea = [TrainFea, ones(Ntrain,1)];

%
parameter_c = 100; %trade-off parameter, c*sum(\ell) + 0.5w'*w
k_factor = [1:1:Ntrain];
k_factor = min(k_factor, Ntrain);

%parameter for algorithm
SGD_options.NumPass = 1000; %outer loop maxIter, for TopK_subgrad
SGD_options.stepSize = 10^-3;    %initial stepsize of subgrad_lr

%save results
Accu = zeros(1, length(k_factor));
w_base = zeros(size(TrainFea,2), length(k_factor));

for k_percentage = 1 : length(k_factor)
    
    ATK_options = struct('c', parameter_c, 'K', min(Ntrain, k_factor(k_percentage))); 
    ATK_options.c = ATK_options.c*Ntrain/ATK_options.K;
    
    [w,~] = logistic_sgd(TrainFea, TrainGnd, ATK_options, SGD_options);
    
    %training accuracy
    margin_train = TrainFea * w; predictLabel = ones(length(TrainGnd),1);
    predictLabel(margin_train < 0) = -1;
    Accu(k_percentage) = sum(double(predictLabel == TrainGnd)) / length(TrainGnd) * 100;
    w_base(:, k_percentage) = w;
    
    %
    fprintf('%-d: %0.2f\n',min(Ntrain, k_factor(k_percentage)), Accu(k_percentage));
end

resfile = [resPath, '.mat'];
[filepath,~,~] = fileparts(resfile);  if ~isdir(filepath), mkdir(filepath), end
save(resfile,'x','y','Accu','k_factor','w_base','w_bayes','bayes_error');

