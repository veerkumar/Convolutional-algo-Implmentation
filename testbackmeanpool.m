X = [ 1 3 5 5 8 6; 2 4 5 6 0 1; 2 4 0 1 3 -1; 3 5 1 2 4 -2];
dzdy = [2 3 1; 1 1 1];

y	= forw_meanpool(X);
dzdx = back_meanpool(X, y, dzdy);

eps = 1.0e-6;
%check dzdx values (deriv of loss with respect to x input values)
fprintf('comparison of analytic and numerical derivs meanpool backprop\n');

dzdxdzdxnumeric = zeros(size(X));
y = forw_meanpool(X);
for i=1:size(X,1)
    for j=1:size(X,2)
        newx = X;
        newx(i,j) = newx(i,j)+eps;
        yprime = forw_meanpool(newx);
        deriv = (yprime-y)/eps;
        %compute dz/dx_ij value using multivariate chain rule        
        dzdxdzdxnumeric(i,j) = dot(deriv(:),dzdy(:));
    end
end

dzdx
dzdxdzdxnumeric

