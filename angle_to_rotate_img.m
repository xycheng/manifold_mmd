function [X ] = angle_to_rotate_img( img, size_w, ts, max_angle)

n = numel(ts);

dim = size_w^2;


X = zeros( n, dim);
for i=1:n
    
    im1 = imrotate( img, ts(i)*max_angle, 'bilinear','crop');
    
    im2=imresize( im1, [size_w, size_w], 'bilinear');
    X(i,:)= im2(:);
end


end