function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];
X1 = sigmoid(X*Theta1');


X2 = [ones(m, 1) X1];

X3 = sigmoid(X2*Theta2');

[~, p] = max(X3, [], 2); % 2 表示在第二个维度取最大值，也就是在每一行取最大值
% [~, p] 其中第一位是最大值的值，第二列是最大值的index
% 最终的结果就是，每一行都有10个分类的可能性，取其中可能性的最大值，而这个可能性所在的列号就是预测的值

% =========================================================================


end
