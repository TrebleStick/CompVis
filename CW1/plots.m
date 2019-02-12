clf
% histo = cell(10,15);

% for i = 1:10
%     for j = 1:15
%         histo{i,j} = histogram(data_train{i,j}, 256); 
%     end
% end

bins = [128 256 512 1024 2048];

for i = 1:5
    [data_train,data_test] = getCalData(bins(i));
    
    subplot(5,3,1+(3*(i-1)))
    bar(data_train(71,:))

    subplot(5,3,2+(3*(i-1)))
    bar(data_train(61,:))


    subplot(5,3,3*i)
    bar(data_train(108,:))
    
  
end
