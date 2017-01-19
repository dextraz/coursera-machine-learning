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





J=0;
Delta_l=0;
for i=1:m,
    % 1. Calculate values of a(1), a(2), a(3) and z(2), z(3)
    a1=X(i,:)';
    a1=[1; a1];
    z2=Theta1*a1;
    a2=sigmoid(z2);
    a2=[1; a2];
    z3=Theta2*a2;
    a3=sigmoid(z3);

    % 2. Reorganize y(i) as a vector yk(k,1)
    yk=zeros(num_labels,1);
    yk(y(i))=1;

    % 3. Compute delta
    delta_3=a3-yk;
    delta_2=((Theta2')*delta_3);
    delta_2 = delta_2(2:end);
    delta_2 = delta_2.*sigmoidGradient(z2);

    % Compute DElta Capital 
    Theta2_grad = Theta2_grad + delta_3*a2';
    Theta1_grad = Theta1_grad + delta_2*a1';

    % Compute sum of J(theta)
    ht=a3;
    s1=-1.*yk.*log(ht);
    s2=-1.*(1.-yk).*log(1.-ht);
    sk=sum(s1.+s2);
    J=J+sk;
end;
J=J/m;

% Cost of regularized J(theta)
T1 = Theta1;
T1(:,1) = 0;   % because we don't add anything for j = 0  
T1_square=T1.*T1;
st1=sum(sum(T1_square));

T2 = Theta2;
T2(:,1) = 0;   % because we don't add anything for j = 0  
T2_square=T2.*T2;
st2=sum(sum(T2_square));

J=J+lambda/(2*m)*(st1+st2);
Theta2_grad=Theta2_grad/m+lambda/m*T2;
Theta1_grad=Theta1_grad/m+lambda/m*T1;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
