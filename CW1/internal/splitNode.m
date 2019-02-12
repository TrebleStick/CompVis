function [node,nodeL,nodeR] = splitNode(data,node,param)
% Split node

visualise = 0;

% Initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];
for n = 1:iter
    
    % Split function - Modify here and try other types of split function
    switch param.funkySplit
        case 'axisAligned' 
            dim = randi(D-1); % Pick one random dimension
            d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
            d_max = single(max(data(:,dim))) - eps;
            t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
            idx_ = data(:,dim) < t;
    
        case 'linear'
            dim = randi(D-1);
            d_min1 = single(min(data(:,dim))) + eps; % Find the data range of this dimension
            d_max1 = single(max(data(:,dim))) - eps;
            t = d_min1 + rand*((d_max1-d_min1)); % Pick a random value within the range as threshold
            
            dim2 = randi(D-1); 
            d_min2 = single(min(data(:,dim2))) + eps; % Find the data range of this dimension
            d_max2 = single(max(data(:,dim2))) - eps;
            t2 = d_min2 + rand*((d_max2-d_min2)); % Pick a random value within the range as threshold
          
            theta = rand*2*pi;
            
            idx_ = cos(theta)*(data(:,dim)-t) + sin(theta)*(data(:,dim2)-t2) > 0;
            
        case 'nonLinear'
            
        case 'twoPixelTest'
        
    
    end
    
    ig = getIG(data,idx_); % Calculate information gain
    
%     if visualise
%         visualise_splitfunc(idx_,data,dim,t,ig,n);
%         pause();
%     end
    
    if (sum(idx_) > 0 && sum(~idx_) > 0) % We check that children node are not empty
        [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
    end
    
end

nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
else
    idx_best = idx_best;
end
end