% https://it.mathworks.com/help/stats/multivariate-normal-distribution.html
mu = [0 0];
%sigma = [0.25 0.3; 0.3 1];
sigma = [1 0; 0 1];
% sigma = [1 0.8; 0.8 4]; %sigma = [1 0.8; 0.8 4];
% sigma = [1 0; 0 4]; %perfettamente centrati vs axes (nell'altra direzione tende a prendere il valor medio)
% sigma = [1 0.8; 0.8 1]; %sghembo vs axes, positive correlation
% sigma = [1 -0.5; -0.5 1]; %sghembo vs axes, negative correlation

x1 = -3:0.2:3;
x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y = mvnpdf(X,mu,sigma);
y = reshape(y,length(x2),length(x1));

surf(x1,x2,y)
caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
axis([-3 3 -3 3 0 0.4])
xlabel('x1')
ylabel('x2')
zlabel('Probability Density')