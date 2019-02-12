clear all; close all;
% Initialisation
init; clc;
% 
% for N = [1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] % Number of trees, try {1,3,5,10, or 20}
%     param.num = N;
%     param.depth = 11;    % trees depth
%     param.splitNum = 10; % Number of trials in split function
%     param.split = 'IG'; % Currently support 'information gain' only
% 
% 
%     % Select dataset
%     [data_train, data_test] = getCalData(256); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}
% 
%     % Train Random Forest
%     fprintf('Training - Num: %i, Depth: %i, Splits: %i\n' , param.num, param.depth, param.splitNum);
%     trees = growTrees(data_train, param);
% 
%     % Test Random Forest
%     testTrees_script;
%     
% end
% %% 4.2 Depth of trees
% 
% init;
% for Dep = [2, 4, 8, 16, 32, 64] % Tree depth, try {2,5,7,11}
%     param.num = 10;
%     param.depth = Dep;    % trees depth
%     param.splitNum = 10; % Number of trials in split function
%     param.split = 'IG'; % Currently support 'information gain' only
% 
%     % Select dataset
%     [data_train, data_test] = getCalData(2048); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}
% 
%     % Train Random Forest
%     fprintf('Training - Num: %i, Depth: %i, Splits: %i\n' , param.num, param.depth, param.splitNum);
%     trees = growTrees(data_train,param);
% 
%     % Test Random Forest
%     testTrees_script;
%     
% end


for Num = [128, 256, 512, 1024, 2048, 4096]
    for Dep = [8]
        for learner = {'linear', 'axisAligned'}
% for Num = [1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
%     for Dep = [2, 4, 8, 16, 32, 64]
            param.num = Num;
            param.depth = Dep;    % trees depth
            param.splitNum = 50; % Number of trials in split function
            param.split = 'IG'; % Currently support 'information gain' only
    %         param.funkySplit = 'axisAligned';
            param.funkySplit = learner{1};

            % Select dataset
            [data_train, data_test] = getCalData(256); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}

            % Train Random Forest
            fprintf('Training - Num: %i, Depth: %i, Splits: %i, Learner: %s\n' , param.num, param.depth, param.splitNum, param.funkySplit);
            trees = growTrees(data_train, param);

            % Test Random Forest
            testTrees_script;
        end
    end
end
%% 5. Experiment with Caltech dataset for image categorisation (Coursework 1)

param.num = 10;
param.depth = 10;    % trees depth
param.splitNum = 3; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

% Complete getData.m by writing your own lines of code to obtain the visual
% vocabulary and the bag-of-words histograms for both training and testing data.
% You can use any existing code for K-means (note different codes require different memory and computation time).

% [data_train, data_test] = getData('Caltech');`
