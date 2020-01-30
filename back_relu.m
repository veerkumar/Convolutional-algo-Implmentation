%back_relu
%{
input x is an mxn matrix
input y is an mxn matrix (output from forward pass)
input dzdy is an mxn matrix of dz/dyij values
output dzdx is an mxn matrix of dz/dxij values
%}

function dzdx = back_relu(x, y, dzdy)
%{ 
    derivative of max(0,x) is 1 if x > 0 else 0
%}
    dydx = double(x >= 0);
   % size(dydx)
    dzdx = dzdy.*dydx;
return