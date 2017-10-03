function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

Cs = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmas = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
error = size(8,8);
%predicts = size(m, size(Cs,2))
disp(size(Cs,1))
for i=1:size(Cs,1)
    for k=1:size(sigmas,1)
        
        model= svmTrain(X, y, Cs(i), @(x1, x2) gaussianKernel(x1, x2, sigmas(k)));
        predictions = svmPredict(model, Xval);
        error(i,k) = mean(double(predictions ~= yval));
        disp('A')
    end
end

[M_sig,I_sig] = min(error,[],1); %row
[M_Cs,I_Cs] = min(error,[],2); %column

[VALUE,Index] = min(M_sig); %sigma
sigma = sigmas(Index(1,1),1);

[VALUE,Index] = min(M_Cs); %C
C = Cs(Index(1,1),1);



% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
