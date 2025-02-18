clc;
close all;
clear;
tic

% 输入数据
data = readmatrix('D:\dataset\cgl\nir_cgl_FULL.xlsx');
shuru = data(3:end,1:end); 
value = readmatrix('D:\dataset\cgl\cgl_value.xlsx');
Y = value(1:end,4);

% 定义变量
SearchAgents = 60;
Max_iterations = 300;
dimension = size(shuru, 2);
lowerbound = zeros(1, dimension); % Lower limit for variables
upperbound = ones(1, dimension); % Upper limit for variables

% 初始化种群
X = rand(SearchAgents, dimension);
X(X < 0.5) = 0; % Convert to binary
X(X >= 0.5) = 1;



% 计算适应度
for i = 1:SearchAgents
    L = X(i,:);
    fit(i) = fobj(shuru, Y,L);
end

%% 迭代优化
for t = 1:Max_iterations
    %% 更新全局最优
    [best, location] = min(fit);
    if t == 1
        PZ = X(location,:); % Optimal location
        fbest = best; % The optimization objective function
    elseif best < fbest
        fbest = best;
        PZ = X(location,:);
    end
    
    %% 阶段1：觅食行为
    for i = 1:SearchAgents
        I = round(1 + rand);
        X_newP1 = X(i,:) + rand(1, dimension) .* (PZ - I .* X(i,:)); % Eq(3)
        X_newP1 = max(X_newP1, lowerbound);
        X_newP1 = min(X_newP1, upperbound);
        
        % 应用Sigmoid函数和二值化
        S = 1 ./ (1 + exp(-10 * (X_newP1 - 0.5))); % Apply Sigmoid
        X_newP1 = double(S >= rand(1, dimension));
        
        % 更新位置
        f_newP1 = fobj(shuru, Y,X_newP1);
        if f_newP1 <= fit(i)
            X(i,:) = X_newP1;
            fit(i) = f_newP1;
        end
    end
    %% 阶段1结束
    
    %% 阶段2：防御策略
    Ps = rand;
    k = randperm(SearchAgents, 1);
    AZ = X(k,:); % 被攻击的斑马
    
    for i = 1:SearchAgents
        if Ps < 0.5
            %% S1：狮子攻击斑马，斑马选择逃跑策略
            R = 0.1;
            X_newP2 = X(i,:) + R * (2 * rand(1, dimension) - 1) * (1 - t / Max_iterations) .* X(i,:); % Eq.(5) S1
            X_newP2 = max(X_newP2, lowerbound);
            X_newP2 = min(X_newP2, upperbound);
        else
            %% S2：其他捕食者攻击斑马，斑马选择进攻策略
            I = round(1 + rand);
            X_newP2 = X(i,:) + rand(1, dimension) .* (AZ - I .* X(i,:)); % Eq(5) S2
            X_newP2 = max(X_newP2, lowerbound);
            X_newP2 = min(X_newP2, upperbound);
        end
        
        % 应用Sigmoid函数和二值化
        S = 1 ./ (1 + exp(-10 * (X_newP2 - 0.5))); % Apply Sigmoid
        X_newP2 = double(S >= rand(1, dimension));
        
        % 更新位置
        f_newP2 = fobj(shuru, Y,X_newP2); % Eq (6)
        if f_newP2 <= fit(i)
            X(i,:) = X_newP2;
            fit(i) = f_newP2;
        end
    end
    
    best_so_far(t) = fbest;
    average(t) = mean(fit);
end

Best_score = fbest;
Best_pos = PZ;
ZOA_curve = best_so_far;
% 结果显示部分
selected_bands_indices = find(Best_pos);


% 剔除未选中的波长变量，仅保留选中的波长变量
Xs_selected = shuru(:, selected_bands_indices);

% 定义存储变量
best_RMSE_test = Inf;
best_results = struct();

% 进行20次建模迭代
for iteration = 1:20
    % Split data into training and prediction sets
    num_samples = size(Xs_selected, 1);
    train_samples = floor(0.7 * num_samples);

    % Generate random indices
    indices = randperm(num_samples);

    % Select training and test sets based on the random indices
    train_indices = indices(1:train_samples);
    test_indices = indices(train_samples + 1:end);

    % Prepare training and test data
    x_train = Xs_selected(train_indices, :);
    y_train = Y(train_indices);
    x_test = Xs_selected(test_indices, :);
    y_test_true = Y(test_indices);

    % Model training
    [~, ~, ~, ~, beta, ~] = plsregress(x_train, y_train, 2 );  % A is the number of latent variables

    % Prediction on training set
    Y_pred_train = [ones(size(x_train, 1), 1) x_train] * beta;
    % Prediction on test set
    Y_pred_test = [ones(size(x_test, 1), 1) x_test] * beta;

    % Calculate Root Mean Square Error for training and test sets
    RMSE_train = sqrt(mse(y_train - Y_pred_train));
    RMSE_test = sqrt(mse(y_test_true - Y_pred_test));

    % Calculate R² for training and test sets
    R2_train = 1 - sum((y_train - Y_pred_train).^2) / sum((y_train - mean(y_train)).^2);
    R2_test = 1 - sum((y_test_true - Y_pred_test).^2) / sum((y_test_true - mean(y_test_true)).^2);

%     % 显示训练和测试集的RMSE和R²
%     disp(['Iteration ', num2str(iteration)]);
%     disp(['训练集上均方根误差: ', num2str(RMSE_train)]);
%     disp(['预测集上均方根误差: ', num2str(RMSE_test)]);
%     disp(['训练集上决定系数: ', num2str(R2_train)]);
%     disp(['预测集上决定系数: ', num2str(R2_test)]);

    % 记录最好的结果
    if RMSE_test < best_RMSE_test
        best_RMSE_test = RMSE_test;
        best_results.RMSE_train = RMSE_train;
        best_results.RMSE_test = RMSE_test;
        best_results.R2_train = R2_train;
        best_results.R2_test = R2_test;
        best_results.selected_bands_indices = selected_bands_indices;
        best_results.beta = beta;
    end
end

% 显示最好的结果
disp('最好的结果:');
disp(['训练集上均方根误差: ', num2str(best_results.RMSE_train)]);
disp(['预测集上均方根误差: ', num2str(best_results.RMSE_test)]);
disp(['训练集上决定系数: ', num2str(best_results.R2_train)]);
disp(['预测集上决定系数: ', num2str(best_results.R2_test)]);
disp('选定的波长带:');
disp(best_results.selected_bands_indices);


% 图形显示部分
figure;
plot(mean(shuru, 1));
hold on;
% plot(selected_bands_indices, mean(shuru(:, selected_bands_indices), 1), 'ro');
% 绘制选中的波长柱状图，设定统一高度
selected_heights = 1 * ones(size(selected_bands_indices));
bar(selected_bands_indices, selected_heights, 'FaceColor', [0.2,0.5,1.0]);
title('Selected Spectral Bands');
xlabel('Wavelength Index');
ylabel('Mean Spectral Intensity');
legend('Original Spectrum', 'Selected Bands');
hold off;

% 收敛曲线图
figure;
plot(ZOA_curve, '-*');
toc
disp(['程序运行时间', num2str(toc)]);


