function [data_train, data_test ] = getCalData( numBins )
    train_name  = strcat('csvs\train_data_', num2str(numBins), '.csv');
    test_name   = strcat('csvs\test_data_' , num2str(numBins), '.csv');
    
    data_train = csvread(train_name);
    data_test = csvread(test_name);
end