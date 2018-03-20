function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X 是 [1 0.5; 1 1.2; xxx] 这样的 Nx2维的
h = X * theta;

% 切记，这里从第二个元素开始
% 这里一个技巧就是把theta的第一个值变成0，这样就没影响了
new_theta = theta;
new_theta(1) = 0;
J = 1/2/m * sum((h-y).^2) + lambda/2/m * sum(new_theta.^2);

grad = 1/m*((h-y)'*X) + lambda/m*new_theta';



% =========================================================================

grad = grad(:);

end