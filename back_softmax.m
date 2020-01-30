%{
input x is an mx1 vector
input y is an mx1 vector (output from forward pass)
input dzdy is an mx1 vector of dz/dyi values
output dzdx is an mx1 vector of dz/dxi values
%}

function dzdx = back_softmax (x,y,dzdy)
   %temp = exp(x);
   %temp_sum = sum(temp);
   %dzdx = dzdy.*(temp/temp_sum - temp^2/temp_sum^2)

    m = size(x,2);
    dydx = zeros(m,m);
    for i= 1:m
        for j = 1:m
            if i ~= j
                numerator = -1*y(j);
            else
                numerator = 1 - y(j);
            end
            dydx(i,j) = y(i)*numerator;
        end    
    end
dzdx_temp=dzdy.*dydx;
dzdx=int16(sum(dzdx_temp));
return