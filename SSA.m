%%
clc;
close all;
clear;
tic
%function [fMin , bestX,Convergence_curve ] = SSA(pop, M,c,d,dim,fobj  )
%% 输入数据
data = readmatrix('D:\dataset\apple\apples_spectral_new.xlsx');
shuru = data(3:end,1:end); 
value = readmatrix('D:\dataset\apple\apples_brix.xlsx');
Y = value(1:end,3);
%% 定义变量
pop = 60;  %种群数量
M = 300; % 迭代次数
%  P_percent = 0.8;    % The population size of producers accounts for "P_percent" percent of the total population size
% pNum = round( pop *  P_percent );    % The population size of the producers

dim = size(shuru, 2);
lb=zeros(1,dim);
ub=ones(1,dim);

Convergence_curve=zeros(1,M);
%%  初始化种群

% 初始化种群
x = randi([0 1], pop, dim);  % 二进制初始化
fit = zeros(pop, 1);
for i = 1:pop
    fit(i) = fobj(shuru, Y, x(i, :));
end
pFit = fit;
pX = x;  % 个体的最优位置
[fMin, bestI] = min(fit);  % 全局最优适应度值
bestX = x(bestI, :);  % 全局最优位置

%% 更新部分
% Start updating the solutions.

% 设置P_percent的初始值和终止值
P_start = 1;
P_end = 0.1;
for t = 1 : M

     % 计算当前迭代的P_percent，使用幂次递减
    P_percent = P_end + (P_start - P_end) * (1 - t / M)^2;
    pNum = round(pop * P_percent);  % 根据P_percent计算生产者数量

    [ ans, sortIndex ] = sort( pFit );% Sort.

    [fmax,B]=max( pFit );
    worse= x(B,:);

    r2=rand(1);
    if(r2<0.8)

        for i = 1 : pNum                                                   % Equation (3)
            r1=rand(1);
            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1*M));
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
            fit( sortIndex( i ) ) = fobj(shuru, Y, x( sortIndex( i ), : ) );
        end
    else
        for i = 1 : pNum

            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )+randn(1)*ones(1,dim);
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
            fit( sortIndex( i ) ) = fobj(shuru, Y, x( sortIndex( i ), : ) );

        end

    end


    [ fMMin, bestII ] = min( fit );
    bestXX = x( bestII, : );


    for i = ( pNum + 1 ) : pop                     % Equation (4)

        A=floor(rand(1,dim)*2)*2-1;

        if( i>(pop/2))
            x( sortIndex(i ), : )=randn(1)*exp((worse-pX( sortIndex( i ), : ))/(i)^2);
        else
            x( sortIndex( i ), : )=bestXX+(abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);

        end
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
        fit( sortIndex( i ) ) = fobj(shuru, Y, x( sortIndex( i ), : ) );

    end
    c=randperm(numel(sortIndex));
    b=sortIndex(c(1:20));
    for j =  1  : length(b)      % Equation (5)

        if( pFit( sortIndex( b(j) ) )>(fMin) )

            x( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) -bestX)));

        else

            x( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);

        end
        x( sortIndex(b(j) ), : ) = Bounds( x( sortIndex(b(j) ), : ), lb, ub );

        fit( sortIndex( b(j) ) ) = fobj(shuru, Y, x( sortIndex( b(j) ), : ) );
    end
    for i = 1 : pop
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end

        if( pFit( i ) < fMin )
            fMin= pFit( i );
            bestX = pX( i, : );


        end
    end

    Convergence_curve(t)=fMin;

end
%% 20次pls建模取最优结果
selected_bands_indices = find(bestX);
% % Display the selected bands
% disp('Selected bands:');
% disp(selected_bands_indices);

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
    [~, ~, ~, ~, beta, ~] = plsregress(x_train, y_train, 6  );  % A is the number of latent variables

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
plot(Convergence_curve,'-*');
toc
disp(['程序运行时间', num2str(toc)]);


%% 定义函数部分
% Application of simple limits/bounds
function s = Bounds(s, Lb, Ub)
    % Apply boundary constraints
    s = max(s, Lb);
    s = min(s, Ub);
    
    % Sigmoid-based binarization
    s = 1 ./ (1 + exp(-s )); % Apply Sigmoid

    % Convert probabilities to binary
    for i = 1:length(s)
        if s(i) >= rand()
            s(i) = 1;
        else
            s(i) = 0;
        end
    end
end




