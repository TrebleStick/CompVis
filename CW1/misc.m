init
% If make.m fails, please check README about detailed instructions.
rng(21)
close all;
imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name} % 10 classes
disp('Loading training images...')
% Load Images -> Description (Dense SIFT)
cnt = 1;
for c = 1:length(classList)
subFolderName = fullfile(folderName,classList{c});
imgList = dir(fullfile(subFolderName,'*.jpg'));
imgIdx{c} = randperm(length(imgList));
imgIdx_tr = imgIdx{c}(1:imgSel(1));
imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));