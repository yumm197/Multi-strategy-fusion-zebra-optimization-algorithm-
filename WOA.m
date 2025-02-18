%% 释放空间
clc
clear
close all
tic
%% 参数设置
SearchAgents_no = 60; % 种群量级
Max_iteration = 300; % 迭代次数

% 数据加载
data = readmatrix('D:\dataset\marzipan\nir_marzipan_FULL.xlsx');
shuru = data(3:end,1:end); 
value = readmatrix('D:\dataset\marzipan\marzipan_value.xlsx');
Y = value(1:end,2);

dim = size(shuru, 2); % 每只鲸鱼的维度
lb = zeros(1, dim);
ub = ones(1, dim);



%% 种群初始化
fitness = zeros(1, SearchAgents_no); % 适应度
Positions = rand(SearchAgents_no, dim); % 初始化位置

% Convert positions to binary (0 or 1)
Positions(Positions < 0.5) = 0;
Positions(Positions >= 0.5) = 1;

for i = 1:SearchAgents_no
    fitness(i) = fobj(shuru, Y,Positions(i,:));
end

[SortFitness, indexSort] = sort(fitness); % 升序，第一个是最小的
% 最优个体
Leader_pos = Positions(indexSort(1), :);
Leader_score = SortFitness(1);

Convergence_curve = zeros(1, Max_iteration); % 收敛曲线

%% 迭代优化
for t = 1:Max_iteration
    
    % 控制参数a
    a = 2 - t * ((2) / Max_iteration);
    % 更新l
    a2 = -1 + t * ((-1) / Max_iteration);
    
    % 参数更新
    for i = 1:size(Positions, 1)
        % A和C更新
        r1 = rand();
        r2 = rand();
        A = 2 * a * r1 - a;
        C = 2 * r2;
        % b和l更新
        b = 1;
        l = (a2 - 1) * rand + 1;
        
        % 更新个体
        for j = 1:size(Positions, 2)
            % 随机数p
            p = rand();
            if p < 0.5
                if abs(A) >= 1
                    % 搜索觅食机制
                    rand_leader_index = randi([1 SearchAgents_no]);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand = abs(C * X_rand(j) - Positions(i, j));
                    Positions(i, j) = X_rand(j) - A * D_X_rand;
                elseif abs(A) < 1
                    % 收缩包围机制
                    D_Leader = abs(C * Leader_pos(j) - Positions(i, j));
                    Positions(i, j) = Leader_pos(j) - A * D_Leader;
                end
            elseif p >= 0.5
                % 螺旋更新位置
                distance2Leader = abs(Leader_pos(j) - Positions(i, j));
                Positions(i, j) = distance2Leader * exp(b * l) * cos(l * 2 * pi) + Leader_pos(j);
            end
        end
        
        % Apply Sigmoid function and binarize
        S = 1 ./ (1 + exp(-10 * (Positions(i, :) - 0.5))); % Apply Sigmoid
        
        for j = 1:dim
            if S(j) >= rand()
                Positions(i, j) = 1;
            else
                Positions(i, j) = 0;
            end
        end
    end
    
    % 越界规范
    for i = 1:size(Positions, 1)
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb; % 超过最大值的设置成最大值，超过最小值的设置成最小值
        fitness(i) = fobj(shuru, Y,Positions(i, :));
        % 最优更新
        if fitness(i) < Leader_score
            Leader_score = fitness(i);
            Leader_pos = Positions(i, :);
        end
    end
    
    Convergence_curve(t) = Leader_score;
end

selected_bands_indices = find(Leader_pos);
WOA_curve = Convergence_curve;

%% 20次建模结果选最优
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
    [~, ~, ~, ~, beta, ~] = plsregress(x_train, y_train,  2  );  % A is the number of latent variables

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

%% 图形显示部分
figure;
plot(mean(shuru, 1));
hold on;
% plot(selected_bands_indices, mean(shuru(:, selected_bands_indices), 1), 'ro');
% 绘制选中的波长柱状图，设定统一高度
selected_heights = 2 * ones(size(selected_bands_indices));
bar(selected_bands_indices, selected_heights, 'FaceColor', [0.2,0.5,1.0]);
title('Selected Spectral Bands');
xlabel('Wavelength Index');
ylabel('Mean Spectral Intensity');
legend('Original Spectrum', 'Selected Bands');
hold off;

% 收敛曲线图
figure;
plot(WOA_curve, '-*');
toc
disp(['程序运行时间', num2str(toc)]);



