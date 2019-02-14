k_values = [128 256 512 1024 2048];
t = [0 0 0 0 0];
for i = 1:5
    t_start = tic;
     [data_train,data_test] = getData('Caltech', k_values(i));
    t(i) = toc - t_start;
end
