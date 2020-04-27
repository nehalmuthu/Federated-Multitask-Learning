function [ primal_obj ] = compute_primal(X, Y, W, Omega, lambda)
% Inputs
%   Xtrain: input training data (m-length cell)
%   Ytrain: output training data (m-length cell)
%   W: current models (d x m)
%   Omega: precision matrix (m x m)
%   lambda: regularization parameter
% Output
%   primal objective

% compute primal
total_loss = 0;
%disp(size(X{1}));
%disp(size(Y{1}));
%disp(size(W(:, 1)));
%class(X{1});
%class(Y{1});
%class(W(:, 1));
%lav = double(Y{1}).*((X{1})*W(:, 1));
%disp(size(lav));

for t=1:length(X)
    preds = double(Y{t}).*((X{t})*W(:, t));
    total_loss = total_loss + mean(max(0.0, 1.0 - preds));
end
primal_obj = total_loss + lambda / 2 * trace(W * Omega * W');

end

