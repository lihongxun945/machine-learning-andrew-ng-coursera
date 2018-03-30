function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% 这个题目的意思是，X每一行都是一个点，然后随机产生了几个centroid ，然后我们返回的 idx 表示x中每一个点对应的最近的 centroid 的序号

for i=1:size(X,1)
  dis = zeros(K, 1);
  for j=1:K
    dis(j,1) = sqrt(sum(power(X(i,:)-centroids(j,:), 2)));
  end
  [~, min_d] = min(dis); % [m, i] = min(xx) 其中m是最小的值，i是这个值的序号
  idx(i, 1) = min_d;
end

% =============================================================

end

