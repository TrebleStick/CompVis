k_values = [128 256 512 1024 2048];

time_sift = zeros(5,1);
time_kmeans = zeros(5,1);
time_quant = zeros(5,1);
for i = 1:5
    disp(i);
    [data_train,data_test, time_sift(i), time_kmeans(i), time_quant(i)] = getData('Caltech', k_values(i));



%     train_name  = strcat('csvs\train_data_', num2str(i), '.csv');
%     test_name   = strcat('csvs\test_data_' , num2str(i), '.csv');
%
%     csvwrite(train_name, data_train);
%     csvwrite(test_name , data_test );
end
