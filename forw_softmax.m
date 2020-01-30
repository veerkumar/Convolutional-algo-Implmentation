%{
input x is mx1 vector
output y is an mx1 vector
    %}

function  y = forw_softmax(x)
    y = exp(x);
    y = y/sum(y);
return