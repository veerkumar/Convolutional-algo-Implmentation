%{
 input x is an 2mx2n matrix
input y is an mxn matrix (output from forward pass)
input dzdy is an mxn matrix of dz/dyij values
output dzdx is an 2mx2n matrix of dz/dxij values   
%}

function dzdx = back_meanpool(x,y,dzdy)
    m = size(x,1)./2;
    n = size(x, 2)./2;
    dzdx  = zeros(size(x)); %create a zero with shape of X
    for i = 1:m 
     for j = 1:n
        for k = 1:4
            k1 = floor((k+1)/2)-1; % find increment in row of 2x2
            k2 = mod(k+1,2);       %find increment in column of 2x2
            dzdx(2*i-1+k1,2*j-1+k2) = 1/4 * dzdy(i,j);  %Store element of 2x2 in actual dX            
        end
     end
    end
return