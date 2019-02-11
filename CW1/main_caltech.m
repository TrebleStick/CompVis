% Main for caltech101 
% This script aims to help the student in understanding the main ideas behind
% random forests and how they can be implemented, being a tool to help the
% on their first coursework on "Selected Topics in Computer Vision" 2018-2019.
%
% The script is divided in 5 different sections as follows:
%
%   1. Data loading/generation
%   2. Random Forest Training
%       2.1 Bagging: Creating subsets of training data
%       2.2 Growing a tree
%           2.2.1 Node splitting
%           2.2.2 Growing the rest of the tree
%           2.2.3 Leaf nodes
%       2.3 Train a random forest
%   3. Inference (test) in random forest
%   4. Random forest parameters
%       4.1 Number of trees
%       4.2 Depth of trees
%   5. Experiment with Caltech dataset for image categorisation (Intro to Coursework 1)
%
% Instructions:
%   - Run the different sections in order (some sections require variables
%   from previous sections)
%   - Try to understand the code and how it relates to theory.
%   - Play with different forest parameters and understand their impact.
%
% The script is based in:
% Simple Random Forest Toolbox for Matlab
%   written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
%   updated by Tae-Kyun Kim, Feb 09, 2017
%   updated by G. Garcia-Hernando, Jan 10, 2018

% Last update: January 2019

% The codes are made for educational purposes only.
% Some parts are inspired by Karpathy's RF Toolbox
% Under BSD Licence

clear all; close all;
% Initialisation
init; clc;

%% 1. Data loading/generation

% Select dataset among {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}

% [data_train, data_test] = getData('Caltech', 256);
[data_train, data_test] = readCalData(256);


%% 2. Random Forest Training

%% 2.1 Bagging: Creating subsets of training data

[N,D] = size(data_train);
frac = 1; % Bootstrap sampling fraction
[labels,~] = unique(data_train(:,end));

%% 2.2 Growing a tree
% Some parameters first
T = 1; % Tree number
param.splitNum = 3; % Number of trials in split function

%% 2.2.1 Node splitting

ig_best = -inf;

for n = 1:param.splitNum
    dim = randi(D-1);                           % Pick one random dimension as a split function
    d_min = single(min(data_train(idx,dim)));   % Find the data range of this dimension
    d_max = single(max(data_train(idx,dim)));
    t = d_min + rand*((d_max-d_min));           % Pick a random value within the range as threshold

    idx_ = data_train(idx,dim) < t;             % Split data with this dimension and threshold

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate Information Gain

    L = data_train(idx_,:);
    R = data_train(~idx_,:);
    H = getE(data_train);        % Calculate entropy
    HL = getE(L);
    HR = getE(R);

    ig = H - sum(idx_)/length(idx_)*HL - sum(~idx_)/length(idx_)*HR

    if ig_best < ig
        ig_best = ig;   % maximu information gain saved
        t_best = t;     % the best threhold to save
        dim_best = dim; % the best split function (dimension) to save
        idx_best = idx_;
    end

    % Visualise the split function and its information gain
    figure(1)
    visualise_splitfunc(idx_,data_train(idx,:),dim,t,ig,0);
    drawnow;
    disp('Press any key to continue');
    pause;
end
% Visualise the best split function saved
visualise_splitfunc(idx_best,data_train(idx,:),dim_best,t_best,ig_best,0);

%% 2.2.2 Growing the rest of the tree
% Let's set some parameters...
param.depth = 5;        % Tree depth
param.split = 'IG';     % Currently support 'information gain' only%

% Initialise base node
trees(T).node(1) = struct('idx',idx,'t',nan,'dim',-1,'prob',[]);
% Split the nodes recursively
for n = 1:2^(param.depth-1)-1
    [trees(T).node(n),trees(T).node(n*2),trees(T).node(n*2+1)] = splitNode(data_train,trees(T).node(n),param);
end

%% 2.2.3 Leaf nodes
% Store class distributions in the leaf nodes
makeLeaf;
% Visualise the class distributions of the first 9 leaf nodes
visualise_leaf;

%% 2.3 Train a random forest
close all;

param.num = 50;         % Number of trees
param.depth = 6;        % Depth of each tree
param.splitNum = 10;     % Number of trials in split function
param.split = 'IG';     % Currently support 'information gain' only

trees = growTrees(data_train,param);

%% 3. Inference (test) in random forest

test_point = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
figure(1)
plot_toydata(data_train);
plot(test_point(:,1), test_point(:,2), 's', 'MarkerSize',20, 'MarkerFaceColor', [.9 .9 .9], 'MarkerEdgeColor','k');

for n=1:4
    figure(2)
    subplot(1,2,1)
    plot_toydata(data_train);
    subplot(1,2,2)
    leaves = testTrees([test_point(n,:) 0],trees);

    % average the class distributions of leaf nodes of all trees
    p_rf = trees(1).prob(leaves,:);
    p_rf_sum = sum(p_rf)/length(trees)

    % visualise the class distributions of the leaf nodes which the data
    % point arrives at (for the first 10 trees)
    for L = 1:10
        subplot(3,5,L); bar(p_rf(L,:)); axis([0.5 3.5 0 1]);
    end
    subplot(3,5,L+3); bar(p_rf_sum); axis([0.5 3.5 0 1]);

    figure(1);
    hold on;
    plot(test_point(n,1), test_point(n,2), 's', 'MarkerSize',20, 'MarkerFaceColor', p_rf_sum, 'MarkerEdgeColor','k');
    pause;
end
hold off;
close all;

% Let's test the RF on our test data

leaves = testTrees_fast(data_test,trees);

for T = 1:length(trees)
    p_rf_all(:,:,T) = trees(1).prob(leaves(:,T),:);
end

p_rf_all = squeeze(sum(p_rf_all,3))/length(trees);

% Let's visualise the results...
visualise(data_train,p_rf_all,[],0);

%% 4. Random forest parameters
%% 4.1 Number of trees

init ;

for N = [1,3,5,10,20] % Number of trees, try {1,3,5,10, or 20}
    param.num = N;
    param.depth = 5;    % trees depth
    param.splitNum = 10; % Number of trials in split function
    param.split = 'IG'; % Currently support 'information gain' only


    % Select dataset
    [data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}

    % Train Random Forest
    trees = growTrees(data_train, param);

    % Test Random Forest
    testTrees_script;

    % Visualise
    visualise(data_train,p_rf,[],0);
    disp('Press any key to continue');
    pause;
end
%% 4.2 Depth of trees

init;
for N = [2,5,7,11] % Tree depth, try {2,5,7,11}
    param.num = 10;
    param.depth = N;    % trees depth
    param.splitNum = 10; % Number of trials in split function
    param.split = 'IG'; % Currently support 'information gain' only

    % Select dataset
    [data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}

    % Train Random Forest
    trees = growTrees(data_train,param);

    % Test Random Forest
    testTrees_script;

    % Visualise
    visualise(data_train,p_rf,[],0);
    disp('Press any key to continue');
    pause;
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
