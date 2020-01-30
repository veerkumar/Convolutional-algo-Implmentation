%{
input x is an 2mx2n matrix (that is, you may assume
that it has an even number of rows and cols)
output y is an mxn matrix


Approch:
Similar to maxpool, 
we will creat a matrix with our 2,2 filter.
take mean of each column,
construct back new matrix(Y) with half the size of privious
%}
function y = forw_meanpool(x)

M = im2col(x, [2 2], 'distinct'); % create new matrix with filter 
M = mean(M); % take mean
y = col2im(M,[1 1],[size(x,1)/2 size(x,2)/2],'distinct');  % construct back new matrix

return