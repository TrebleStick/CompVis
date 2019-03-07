clf
% histo = cell(10,15);

% for i = 1:10
%     for j = 1:15
%         histo{i,j} = histogram(data_train{i,j}, 256); 
%     end
% end

trees = [5 10];
depths = [6 9];

[data_train, data_test] = getRFData(5, 6);

subplot(4,3,1)
bar(data_train(71,:))

subplot(4,3,2)
bar(data_train(61,:))

subplot(4,3,3)
bar(data_test(108,:))

[data_train, data_test] = getRFData(10, 6);

subplot(4,3,4)
bar(data_train(71,:))

subplot(4,3,5)
bar(data_train(61,:))

subplot(4,3,6)
bar(data_test(108,:))

[data_train, data_test] = getRFData(5, 9);

subplot(4,3,7)
bar(data_train(71,:))

subplot(4,3,8)
bar(data_train(61,:))

subplot(4,3,9)
bar(data_test(108,:))

[data_train, data_test] = getRFData(10, 9);

subplot(4,3,10)
bar(data_train(71,:))

subplot(4,3,11)
bar(data_train(61,:))

subplot(4,3,12)
bar(data_test(108,:))




