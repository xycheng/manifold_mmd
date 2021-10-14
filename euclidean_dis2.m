function [D2]=euclidean_dis2(X,Y)
dis2XX=squareform(pdist(X).^2);
dis2YY=squareform(pdist(Y).^2);
dis2XY=pdist2(X,Y).^2;
D2=[dis2XX, dis2XY; dis2XY', dis2YY];
end
