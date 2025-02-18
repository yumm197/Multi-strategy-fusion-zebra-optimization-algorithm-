% 定义适应度函数
function fitness = fobj(shuru, Y, selected_bands)
    % 剔除未选中的波长变量
    selected_shuru = shuru(:, selected_bands == 1);
%     selected_shuru = selected_bands .* shuru;
    % Split data into training and prediction sets
    num_samples = size(selected_shuru, 1);
    train_samples = floor(0.7 * num_samples);

    % Generate random indices
    indices = randperm(num_samples);

    % Select training and test sets based on the random indices
    train_indices = indices(1:train_samples);
    test_indices = indices(train_samples + 1:end);

    % Prepare training and test data
    x_train = selected_shuru(train_indices, :);
    y_train = Y(train_indices);
    x_test = selected_shuru(test_indices, :);
    y_test_true = Y(test_indices);

    % 计算潜变量数量
    ncomp = min( size(x_train,2), 6 );  % 设置最大值为 10
 
    % Model training
    [~, ~, ~, ~, beta, ~] = plsregress(x_train, y_train , ncomp ); % A is the number of latent variables

    % Prediction
    Y_pred_test = [ones(size(x_test, 1), 1) x_test] * beta;

    % Calculate Root Mean Square Error
    RMSE_test = sqrt(mse(y_test_true - Y_pred_test));

    % Fitness value is the RMSE on the test set
    fitness = RMSE_test;
end