k_values = [128,256,1024,2048,4096];

for i = k_values
    [data_train,data_test,data_book] = getData('Caltech', i);
    train_name  = strcat('csvs\train_data_', num2str(i), '.csv');
    test_name   = strcat('csvs\test_data_' , num2str(i), '.csv');
    book_name   = strcat('csvs\code_book_' , num2str(i), '.csv');
    csvwrite(train_name, data_train);
    csvwrite(test_name , data_test );
    csvwrite(book_name , data_book );
end