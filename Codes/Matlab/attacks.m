function [I_mean,I_median,I_noise,I_rotate,I_crop,I_jpeg,I_shear] = attacks(I)

[a,b,c] = size(I);


I_mean = double(imboxfilt(I,3));

I_median = I;
for i=1:3
    I_median(:,:,i) = double(medfilt2(I(:,:,i)));
end

x = tand(20);
T = affine2d( [1 0 0; x 1 0; 0 0 1] );
I_shear = imresize(double(imwarp(I,T,'cubic','FillValues',0)), [a b]);

I_noise = double(imnoise(uint8(I),"gaussian",0,1e-3));

I_rotate = double(imrotate(I,45,"crop"));

e = 10;
I_crop = double(imresize(imcrop(I,[b/e,a/e,b-2*b/e,a-2*a/e]),[a b]));

imwrite(uint8(I), 'temp.jpg');
I_jpeg = double(imread("temp.jpg"));
delete temp.jpg;

end

