%forward maxpool
%{
Function y = forw_maxpool(x)
input x is an 2mx2n matrix (that is, you may assume
that it has an even number of rows and cols)
output y is an mxn matrix 
%}
function ymat = forw_maxpool(x)

M = im2col(x,[2 2],'distinct'); % taking 2X2 and making it column in M matrix
M = max(M); % Find max of each row
ymat = col2im(M,[1 1],[size(x,1)/2 size(x,2)/2],'distinct');  %Rearrage it in new matrix with size/2 as dimension

return