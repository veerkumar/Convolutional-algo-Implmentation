%forward relu 
% Function y = forw_relu(x)
% input x is an mxn matrix
% output y is an mxn matrix , which is activated with Relu function
function ymat = forw_relu(x)

ymat = max(x,0);

return