% function output = test()
% 
%     function out = get_eem(x) 
%          out = cent_tr{x};
%     end
% 
%     parfor i = 1:10
%         for j = 1:15
%     %         data_train{i,j} = cent_tr(data_train_idx{i,j});
%             data_train{i,j} = arrayfun(@get_eem, data_train_idx{i,j});
%         end
%     end
%     output = data_train;
% end


parfor i = 1:10
    for j = 1:15
        data_train{i,j} = cent_tr(data_train_idx{i,j},:);
    end
end