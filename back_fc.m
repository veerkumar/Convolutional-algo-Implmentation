%{
input x is an mxn matrix
input w is an mxn matrix of weights
b is a scalar bias value
input y is a scalar value (output from forward pass)
input dzdy is a scalar value dx/dy
output dzdx is an mxn matrix of dz/dxij values
output dzdw is an mxn matrix of dz/dwij values
output dzdb is a value dz/db
%}

function [dzdx, dzdw, dzdb] = back_fc(x, w, b, y, dzdy)
    dzdx = dzdy*w; % derivative of y with respect to x is jusT "w"
    dzdw = dzdy*x; % derivative of x with respect to w is just "X"
    dzdb = dzdy;   % derivative of b with respec to b is just 1
end