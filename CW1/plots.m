clf
% histo = cell(10,15);

% for i = 1:10
%     for j = 1:15
%         histo{i,j} = histogram(data_train{i,j}, 256); 
%     end
% end

subplot(2,2,1)
bar(data_train(1,:))

subplot(2,2,2)
bar(data_train(23,:))


subplot(2,2,3)
bar(data_train(60,:))


subplot(2,2,4)
bar(data_train(75,:)) 
