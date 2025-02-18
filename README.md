# Introduction
- The codebase contains matlab code for the GWO, WOA, SSA, ZOA, and MFZOA algorithms as well as corn and soil datasets.

# Usage
Using the MFZOA algorithm as an example, open the main program for the algorithm code in `ZOA.m`
## Input
You can modify the dataset file path in the following code to ensure that the data is read correctly.
```matlab
data=readmatrix('D:\dataset\corn\nir_corn_FULL.xlsx');
shuru = data(3:end,1:end); 
value = readmatrix('D:\dataset\corn\corn_value.xlsx');
Y = value(1:end,3);
```
## Output
- `RMSE_train`:Root mean square error on the training set.
- `RMSE_test`:Root mean square error on the test set.
- `R2_train`:Coefficient of determination on the training set.
- `R2_test`:Coefficient of determination on the test set.
- `selected_bands_indices`:Characteristic wavelengths filtered by the algorithm.
## Notation
You can change the hyperparameter settings of the algorithm at any time (e.g. population size, number of iterations).
```MATLAB
SearchAgents= 60 ;
Max_iterations= 300 ;
dimension= size(shuru, 2) ;
lowerbound=zeros(1,dimension);                              % Lower limit for variables
upperbound=ones(1,dimension);                              % Upper limit for variables
```
Example 1 : multi-strategy fusion of zebra optimization algorithm (MFZOA)
```MATLAB
SearchAgents= 60 ;
Max_iterations= 300 ;
dimension= size(shuru, 2) ;
lowerbound=zeros(1,dimension);                              % Lower limit for variables
upperbound=ones(1,dimension);                              % Upper limit for variables

%% INITIALIZATION

    X =  rand(SearchAgents, dimension) < 0.5;                   % Initial population
for i =1:SearchAgents
    L=X(i,:);
    fit(i)=fobj(shuru, Y,L);
end
%%
for t=1:Max_iterations
    %% update the global best (fbest)
    [best , location]=min(fit);
    if t==1
        PZ=X(location,:);                                           % Optimal location
        fbest=best;                                           % The optimization objective function
    elseif best<fbest
        fbest=best;
        PZ=X(location,:);
    end
  %% PHASE1: Foraging Behaviour
    for i=1:SearchAgents
        
        I=round(1+rand);
        X_newP1=X(i,:)+ rand(1,dimension).*(PZ-I.* X(i,:)); %Eq(3)
         X_newP1( X_newP1<0.5)=0; X_newP1( X_newP1>=0.5)=1;
%         X_newP1= max(X_newP1,lowerbound);X_newP1 = min(X_newP1,upperbound);
        
        
        % Updating X_i using (5)
        f_newP1 = fobj(shuru, Y,X_newP1);
        if f_newP1 <= fit (i)
            X(i,:) = X_newP1;
            fit (i)=f_newP1;
        end

    end
    %% End Phase 1: Foraging Behaviour
    
    %% PHASE2: defense strategies against predators
    Ps=2*cos(i/Max_iterations)-1;
    k=randperm(SearchAgents,1);
    AZ=X(k,:);% attacked zebra
    
    for i=1:SearchAgents
        
        if Ps<0.5
            %% S1: the lion attacks the zebra and thus the zebra chooses an escape strategy
            R=0.1;
            X_newP2= X(i,:)+ R*(2*rand(1,dimension)-1)*(2*cos(pi*i/3*Max_iterations)-1).*X(i,:);% Eq.(5) S1
            X_newP2(X_newP2<0.5)=0;X_newP2(X_newP2>=0.5)=1;
%             X_newP2= max(X_newP2,lowerbound);X_newP2 = min(X_newP2,upperbound);
      
        else
            %% S2: other predators attack the zebra and the zebra will choose the offensive strategy
            
            I=round(1+rand(1,1));
            X_newP2=X(i,:)+ rand(1,dimension).*(AZ-I.* X(i,:)); %Eq(5) S2
            X_newP2(X_newP2<0.5)=0;X_newP2(X_newP2>=0.5)=1;
%             X_newP2= max(X_newP2,lowerbound);X_newP2 = min(X_newP2,upperbound);
             
        end
        
        f_newP2 = fobj(shuru, Y,X_newP2); %Eq (6)
        X_newP2P = f_newP2 + f_newP2 .* (i/Max_iterations).*normpdf(f_newP2,0,1);
        X_newP2P(X_newP2P<0.5)=0;X_newP2P(X_newP2P>=0.5)=1;
         if f_newP2 <= fit (i)
             X(i,:) = X_newP2;
             fit (i)=f_newP2;
         end
        new_fitness_P2P = fobj(shuru, Y,X_newP2P);
         if new_fitness_P2P<f_newP2
             X_newP = X_newP2P;
             fit (i)= new_fitness_P2P;
         else
            X_newP = X_newP2;
            fit (i)=f_newP2;
         end
    end 
   
    best_so_far(t)=fbest;
    average(t) = mean (fit);
    
end 

Best_score=fbest;
Best_pos=PZ;
selected_bands_indices = find(Best_pos);

Xs_selected = shuru(:, selected_bands_indices);

best_RMSE_test = Inf;
best_results = struct();

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
    [~, ~, ~, ~, beta, ~] = plsregress(x_train, y_train, 10 );  % A is the number of latent variables

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
```
# Requirement
- MATLAB 2014 or above
- Statistics and Machine Learning Toolbox
