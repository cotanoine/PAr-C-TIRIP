clear
addpath("Codes/Matlab");

c0=1;c1=1;c2=1;c3=1;
A = [ 1      c0          c1;
      c2    1+c0*c2    c1*c2;
      c3 c0*c1*c2*c3 1+c1*c3];
M = 255;

f = 3.6;
d_0 = 0.5;

alpha = 0.04;
beta = 0.02;

I = imread("Images/article/1.png");
W = imread("Images/article/55.png");
Mask = imread("Images/article/m.png");

[a,b,c] = size(I);



I_haar = round(apply_haar(I,W,alpha));
W_prime = reverse_haar(I,I_haar,alpha);

imshow([W,W_prime])
ssim(W,W_prime)