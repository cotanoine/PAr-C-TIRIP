function [I_mean,I_median,I_noise,I_rotate,I_crop,I_jpeg,I_shear] = attacks(I)

[a,b,c] = size(I);


I_mean = imboxfilt(I,3);

I_median = I;
for i=1:3
    I_median(:,:,i) = medfilt2(I(:,:,i));
end

x = tand(20);
T = affine2d( [1 0 0; x 1 0; 0 0 1] );
I_shear = imresize(imwarp(I,T,'cubic','FillValues',0), [a b]);

I_noise = imnoise(I,"gaussian",0,1e-3);

I_rotate = imrotate(I,45,"crop");

e = 10;
I_crop = imresize(imcrop(I,[b/e,a/e,b-b/e,a-a/e]),[a b]);

imwrite(uint8(I), 'temp.jpg');
I_jpeg = double(imread("temp.jpg"));
delete temp.jpg;

end

