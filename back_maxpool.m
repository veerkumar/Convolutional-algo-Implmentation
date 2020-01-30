%back_maxpool
%{
Function y = forw_maxpool(x)
input x is an 2mx2n matrix (that is, you may assume
that it has an even number of rows and cols)
output y is an mxn matrix
%}

function dzdx = back_maxpool(x, y, dzdy)
%{
    derivative of maxpool: replace position of max element in Y with value
    from dzdy
    
    Approch, 
    Create a zeroed array,
    find position of max element in the original matrix, fill dzdy element
    based on position of max element in X.
    convert back tempory matrix back to original X shape
%}

M = im2col(x,[2 2],'distinct');
temp = zeros(size(M));

[max_item, index] = max(M);

for i = 1:size(M,2)
    temp(index(i),i)=dzdy(i);   %fill dzdy at respective postion.
end

dzdx = col2im(temp,[2,2],size(x),'distinct'); %convert to original
return