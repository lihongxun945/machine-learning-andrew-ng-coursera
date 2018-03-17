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

% 这里 nn_params 是提前准备的一些随机值，然后我们用来初始化我们的两个参数 Theta1 和 Theta2
% 其中 num_labels 其实就是 output layer size

% 得到一个 hidden_layer_size * (input_layer_size + 1) 层的 矩阵
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
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

% part 1 是执行神经网络的计算，算出最终的值

% hidden层的输出值
v2 = sigmoid(Theta1 * [ones(m, 1) X]'); % hiddenx401 * 401x5000，结果是 hidden x 5000 维度的矩阵

% output 层的输出值
h_theta = sigmoid(Theta2 * [ones(m, 1) v2']');  % 5000 x hidden+1 * (hidden+1 x 5000) 结果是 5000x5000

% 本来y是 5000*1 维数组，每一行都是一个 1~10 的数字，而神经网络中每一个数字要转成一个只有 0,1 的数字来表示
y_new = zeros(num_labels, m); % 10*5000
for i=1:m,
  y_new(y(i),i)=1;
end

% 为什么sum 两次？ 因为括号内的结果是 10x5000 * 5000x5000 结果是 10x5000 的矩阵，所以需要sum两次
J = sum(sum(-y_new .* log(h_theta) - (1-y_new).*log(1-h_theta)))/m

% 不要算第一个
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% 加上公式末尾的那个

Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

J += Reg

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

% 抄的
% 计算过程是：
% 1. 先把我们预测的结果计算出来
% 2. 计算出预测结果和正确结果的差值
% 3. 反向更新 theta
for t=1:m

    % Step 1
	a1 = X(t,:); % X already have a bias Line 44 (1*401)
  a1 = [1 a1];
	z2 = Theta1 * a1'; % (25*401)*(401*1)
	a2 = sigmoid(z2); % (25*1)
    
  a2 = [1 ; a2]; % adding a bias (26*1)
	z3 = Theta2 * a2; % (10*26)*(26*1)
	a3 = sigmoid(z3); % final activation layer a3 == h(theta) (10*1)
    
    % Step 2
	delta_3 = a3 - y_new(:,t); % (10*1)
	
  z2=[1; z2]; % bias (26*1)
    % Step 3
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % ((26*10)*(10*1))=(26*1)

    % Step 4
	delta_2 = delta_2(2:end); % skipping sigma2(0) (25*1)

	Theta2_grad = Theta2_grad + delta_3 * a2'; % (10*1)*(1*26)
	Theta1_grad = Theta1_grad + delta_2 * a1; % (25*1)*(1*401)
    
end;

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% 
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
% 
% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m; % for j = 0
% 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
