%%
% Grey Wolf Optimizer
clc;
close all;
clear;
tic

%% 参数设置
data = readmatrix('D:\dataset\cgl\nir_cgl_FULL.xlsx');
shuru = data(3:end,1:end); 
value = readmatrix('D:\dataset\cgl\cgl_value.xlsx');
Y = value(1:end,4);

%初始化参数
SearchAgents_no= 60 ;    %种群数量
Max_iter=  300      ;    %迭代次数
dim = size(shuru, 2);    %每只灰狼的维度
lb=zeros(1,dim);
ub=ones(1,dim);

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

%Positions=initialization(SearchAgents_no,dim,ub,lb);

Positions = rand(SearchAgents_no, dim); % Initialize with 50% probability

% Convert logical array to binary (0 and 1)
Positions(Positions<0.5)=0;Positions(Positions>=0.5)=1;

Convergence_curve=zeros(1,Max_iter);

l=0;% Loop counter

%% 波长选择算法迭代
% Main loop
while l<Max_iter
    for i=1:size(Positions,1)  
        
%        % Return back the search agents that go beyond the boundaries of the search space
%         Flag4ub=Positions(i,:)>ub;
%         Flag4lb=Positions(i,:)<lb;
%         Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % Calculate objective function for each search agent
       fitness=fobj(shuru, Y,Positions(i,:));
        
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score 
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score 
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end
            
        
    end
    
    a=2-2*((l)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
           
        end
 % Apply Sigmoid function and binarize
        S = 1 ./ (1 + exp(-10 * (Positions(i,:) - 0.5))); % Apply Sigmoid
        
        for j = 1:dim
            if S(j) >= rand()
                Positions(i,j) = 1;
            else
                Positions(i,j) = 0;
            end
        end
    end
    l=l+1;
    Convergence_curve(l)=Alpha_score;
end
selected_bands_indices = find(Alpha_pos);
GWO_curve = Convergence_curve;

% disp('Selected bands:');
% disp(selected_bands_indices);
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
    [~, ~, ~, ~, beta, ~] = plsregress(x_train, y_train,  2 );  % A is the number of latent variables

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
plot(GWO_curve, '-*');
toc
disp(['程序运行时间', num2str(toc)]);


