function [data_train, data_test ] = getCalData( numBins )
    train_name  = strcat('csvs\train_data_', num2str(numBins), '.csv');
    test_name   = strcat('csvs\test_data_' , num2str(numBins), '.csv');
    
    data_train_test = csvread(train_name);
    data_test_test = csvread(test_name);
%     folderName = './Caltech_101/101_ObjectCategories';
%     classList = dir(folderName);
%     classList = {classList(3:end).name} % 10 classes
    
    
    data_train = zeros(150, numBins+1);
    data_test  = zeros(150, numBins+1);
    for i = 1:10
        for j = 1:15
            data_train(((i-1)*15)+j,:) = [data_train_test(((i-1)*15)+j,:) i];  
            data_test(((i-1)*15)+j,:) = [data_test_test(((i-1)*15)+j,:) i]; 
        end
    end
end