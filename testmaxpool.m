% We need to compute dzdx both analytically and numerically.
% The test will be successful if both yield same result
X = [ 1 3 5 5 8 6; 2 4 5 6 0 1; 2 4 0 1 3 -1; 3 5 1 2 4 -2];
dzdy = [2 3 1; 1 1 1];

%forward pass to compute Y
Y = forw_maxpool(X);

%computing the backprop derivatives analytically 
dzdx=back_maxpool(X,Y,dzdy);

%now compute them by using numerical derivatives 

% numerically compute dz/dw
eps = 1.0e-6;
dzdxnumeric = zeros(size(X)./2);
Y = forw_maxpool(X);
for i=1:size(X,1)
    for j=1:size(X,2)
        filt = X;
        filt(i,j) = filt(i,j)+eps;
        yprime = forw_maxpool(filt);
        deriv = (yprime-Y)/eps;
        dzdxnumeric(i,j) = dot(deriv(:),dzdy(:));
    end
end
%we will just compare them by eye
%this could be more fancy, like computing max abs diff between the two
fprintf('comparison of analytic and numerical derivs maxpool backprop\n');
fprintf('comparing dz/dx values\n');
dzdx
dzdxnumeric