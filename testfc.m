%make up some numbers for input array X
x = [ -1 -2 4 5 8; 2 4 -1 6 0; 2 4 0 1 3];

%make up some numbers for dz/dY coming in from backprop 
w = [2 4 1 3 2; 3 1 1 3 2; 1 2 1 5 1];

bias = 3;
dzdy = 8;

y = forw_fc(x, w, bias);

%Analytical method for backprop
[dzdx, dzdw, dzdb] = back_fc(x,w,bias,y,dzdy);

%check dzdw values (deriv of loss with respect to filter values)
fprintf('comparison of analytic and numerical derivs Fully connect backprop\n');
eps = 1.0e-6;
dzdwnumeric = zeros(size(w));
y = forw_fc(x,w,bias);
for i=1:size(w,1)
    for j=1:size(w,2)
        filt = w;
        filt(i,j) = filt(i,j)+eps;
        yprime = forw_fc(x,filt,bias);
        deriv = (yprime-y)/eps;
        %compute dz/dw_ij value using multivariate chain rule
        %   deriv contains all dy/dw_ij values and dzdy contains all dz/dy values
        %   so to compute dz/dw_ij we sum up all products (dz/dy_kl)*(dy_kl/dw_ij) 
        dzdwnumeric(i,j) = dot(deriv,dzdy(:));
    end
end
dzdw
dzdwnumeric


%check dzdx values (deriv of loss with respect to x input values)
fprintf('Numerical gradient check for dzdx: \n')
eps = 1.0e-6;
dzdxnumeric = zeros(size(x));
y = forw_fc(x,w,bias);
for i=1:size(x,1)
    for j=1:size(x,2)
        newx = x;
        newx(i,j) = newx(i,j)+eps;
        yprime = forw_fc(newx,w,bias);
        deriv = (yprime-y)/eps;
        %compute dz/dx_ij value using multivariate chain rule        
        dzdxnumeric(i,j) = dot(deriv(:),dzdy(:));
    end
end
dzdx
dzdxnumeric


%check dzdb value(deriv of loss with respect to bias value)
fprintf('Numerical gradient check for dzdb: \n')
eps = 1.0e-6;
y = forw_fc(x,w,bias);
yprime = forw_fc(x,w,bias+eps);
bderiv = (yprime-y)/eps;
dzdbnumeric = sum(dot(deriv(:),dzdy(:)));
dzdb
dzdbnumeric
