%% Task 3.3
clc;
clear;
close all;

%% Load Data
data = load("data33.mat");
X = data.X;
figure()
scatter(X(1,1:100), X(2,1:100), 50, 'blue', 'filled')
hold on;
scatter(X(1,101:end), X(2,101:end), 50, 'red', 'filled')
title('Initial Points')
xlabel('x')
ylabel('y')
legend('Class 1', 'Class 2')

%% K-means Algorithm
K = 2; % Set the amount of Z's
iter = 25; % Set the amount of iterations
[C, Z] = KMeans(X,K,iter); % Run the K-means algorithm
Class_Err = classificationError(X,Z); % Calculate the classification error
disp(['Total classification error: ' num2str(Class_Err * 100) '%'])

%% Optimize K-means
X_Norm = vecnorm(X,2,1);
X_New = [X ; X_Norm.^2];
[C_b, Z_b] = KMeans(X_New,K,iter); % Run the K-means algorithm
Class_Err_Opt = classificationError(X_New,Z_b); % Calculate the classification error
disp(['Total classification error: ' num2str(Class_Err_Opt * 100) '%'])

%% Plot Results and Data
Plot_points = cell(1,K);
Plot_points_opt = cell(1,K);

for i=1:size(X,1)
    Class_ind = C{i};
    Class_ind_opt = C_b{i};
    Class_points = X(:, Class_ind);
    Class_points_opt = X(:,Class_ind_opt);
    Plot_points{i} = Class_points;
    Plot_points_opt{i} = Class_points_opt;
end

%===========PLOTS==========%
figure()
scatter(X(1,1:100), X(2,1:100), 50, 'blue', 'filled')
hold on;
scatter(X(1,101:end), X(2,101:end), 50, 'red', 'filled')
hold on;
scatter(Plot_points{1}(1,:), Plot_points{1}(2,:), 15, 'cyan', 'filled')
hold on;
scatter(Plot_points{2}(1,:), Plot_points{2}(2,:), 15, 'yellow', 'filled')
title('Initial & predicted class points')
xlabel('x')
ylabel('y')
legend('Class 1','Class 2', 'Predicted Class 1', 'Predicted Class 2')

%===========PLOTS==========%
figure()
scatter(X(1,1:100), X(2,1:100), 50, 'blue', 'filled')
hold on;
scatter(X(1,101:end), X(2,101:end), 50, 'red', 'filled')
hold on;
scatter(Plot_points_opt{1}(1,:), Plot_points_opt{1}(2,:), 15, 'cyan', 'filled')
hold on;
scatter(Plot_points_opt{2}(1,:), Plot_points_opt{2}(2,:), 15, 'yellow', 'filled')
title('Initial & predicted class points after adding one more dimention to X')
xlabel('x')
ylabel('y')
legend('Class 1','Class 2', 'Predicted Class 1', 'Predicted Class 2')

%% Functions
function [C, Z] = KMeans(X, K, iterations)
    % Choose randomly Z
    sz = [size(X,1), K];    
    Z = zeros(sz);    
    maxi = size(X,2);
    all_distances = zeros([1,iterations]);   
    
    for i=1:K % Initialize Centers
        center_ind = randi(maxi);
        center = X(:,center_ind);
        Z(:,i) = center ;
    end
        
    for iter=1:iterations % Run the K-means algorithm
        distance = 0;
        C = cell(1, K);     

        for i=1:K % Initialize C
            C{i} = [];
        end       

        for i=1:maxi % Classify data
            [~ , index] = min(vecnorm(Z'-X(:,i)',2,2));
            C{index}(end + 1) = i;
        end     

        for i=1:K % Change the centers
            class_indecies = C{i};
            class_points = X(:, class_indecies);
            distance = distance + sum(vecnorm(Z(:,i)' - class_points', 2, 2)); 
            Z(:,i) = mean(class_points,2);            
        end
        all_distances(iter) = distance;
    end
    disp('%======K-means Result======%')
end

function err = classificationError(X,Z)
    tot_err_1 = 0;
    tot_err_2 = 0;
    maxi = size(X,2);

    for i=1:maxi
        [~ , index] = min(vecnorm(Z'-X(:,i)',2,2));
        if index == 1 && i >=100
            tot_err_1 = tot_err_1 + 1;

        elseif index == 2 && i<100
            tot_err_2 = tot_err_2 + 1;
        end
    end    
    err = (tot_err_1 + tot_err_2)/maxi;
end
