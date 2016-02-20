function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1); % K

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % result set

% Add ones to the X data matrix (bias unit)
X = [ones(m, 1) X];

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

m_result = sigmoid(X * Theta1'); % Layer 1 predication
m_result = [ones(m, 1) m_result]; % Add ones to the m_result data matrix

p_result = sigmoid(m_result * Theta2'); % Layer 2 predication
[value, p] = max(p_result, [], 2); % max values in value, index in p

% =========================================================================

end
