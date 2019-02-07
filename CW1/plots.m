clf
% histo = cell(10,15);

% for i = 1:10
%     for j = 1:15
%         histo{i,j} = histogram(data_train{i,j}, 256); 
%     end
% end

subplot(2,2,1)
histogram(data_train{1,1}, 256)

subplot(2,2,2)
histogram(data_train{2,2}, 256)


subplot(2,2,3)
histogram(data_train{3,8}, 256)


subplot(2,2,4)
histogram(data_train{4,1}, 256) 
