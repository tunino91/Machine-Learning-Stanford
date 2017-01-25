function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
bias = 1;         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%% =========== Part 1: Forward Propogate all the Examples one time at once! =============

X = [ones(m, 1) X];
z_2 = Theta1 * X';
a_2 = sigmoid(z_2);
a_2 = [ones(1, m); a_2];
z_3 = Theta2 * a_2;               % 10 x 5000
Hx_allExamples = sigmoid(z_3);    % 10 x 5000 ---> h(x)
a_4 = Hx_allExamples';            % a_4: 5000 x 10
[p,I] = max(a_4, [],2) % Return the max of the each row and their indices i.e.Take the highest confidence)
p = I;                 % p is the prediction of the network that is h(x), each row is a result of one example. [5;3;5;8;2;..]

%% =========== Ground truths are in one hot shot form. yk = 10 x 5000 ===========

yk=zeros(num_labels,m);
for i=1:m
    yk(y(i,1),i)=1;
end

%% =========== Part 2: Cost Function Implementation--No Regularization-- =============
% >>> Insert Cost Function Without Reg <<< %
% Implemented function:  -1/m*((log(Hx_allExamples)).*yk + (1.-yk).*(log(1.-Hx_allExamples)))
one = ones(num_labels,m); 
x = log(Hx_allExamples);
r = log(one - Hx_allExamples);
forAllUnits = (-x.*yk - (one-yk).*(r));
allUnits = sum(forAllUnits,1);
allExamples = sum(allUnits,2);

J = (1/m)*allExamples; % Cost Function without regularization

%% =========== Part 3: Cost Function Implementation--Regularization-- =============

theta_1 = Theta1;
theta_1(:,1) = []; % Take out the first column, that is the biases for the first layer
theta_2 = Theta2;
theta_2(:,1) = []; % Take out the first column, that is the biases for the second layer

% >>> Insert Regularization Cost Part <<< %
mostInner_part_1 = sum((theta_1).^2, 2);
inner_part_1 = sum (mostInner_part_1, 1);

mostInner_part_2 = sum((theta_2).^2, 2);
inner_part_2 = sum(mostInner_part_2, 1);

reg = lambda/(2*m)*(inner_part_1 + inner_part_2);

J = J + reg; % Cost Function with regularization

%% =========== Part 4: Backpropogation Implementation =============

for i=1:m
    %%% Forward propogate one image at a Time %%%
    a1_withBias = X(i,:)'; % a1_withBias: 401 x 1
    a1 = a1_withBias(2:end);
   
    z_2 = Theta1 * a1_withBias;       % 25 x 1
    a_2 = sigmoid(z_2);               % 25 x 1
    a_2_withBias = [bias; a_2];       % 26 x 1 
    
    z_3 = Theta2 * a_2_withBias;      % 10 x 1
    hx_oneExample = sigmoid(z_3);     % 10 x 1 ---> h(x)
    
    error_3 = hx_oneExample - yk(:,i); 
    
    %%% Take out the first column, that is the biases for the second layer
    theta_2 = Theta2; 
    theta_2(:,1) = []; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    error_2 = theta_2' * error_3 .* sigmoidGradient(z_2);
    
    %%% Gradient Calculation %%%
    Theta2_grad = Theta2_grad + error_3*(a_2_withBias)'
    Theta1_grad = Theta1_grad + error_2*(a1_withBias)'
    
end

%% Total Gradients after the forward prop. of all examples %%
Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;

%% Total Gradients after the forward prop. of all examples--Accounted for Regularization %%

theta_2 = Theta2; 
theta_2(:,1) = [];

Theta2_Add = (lambda/m) .* theta_2;
Theta2_Add = [zeros(num_labels, 1) Theta2_Add];
Theta2_grad = Theta2_Add + Theta2_grad; 

theta_1 = Theta1; 
theta_1(:,1) = [];

Theta1_Add = (lambda/m) .* theta_1;
Theta1_Add = [zeros(hidden_layer_size, 1) Theta1_Add];
Theta1_grad = Theta1_Add + Theta1_grad; 

% -------------------------------------------------------------

% =========================================================================


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
