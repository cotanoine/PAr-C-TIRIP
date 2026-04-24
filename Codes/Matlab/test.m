clear
addpath("Codes/Matlab");

c0=0;c1=0;c2=0;c3=0;
A = [ 1      c0          c1;
      c2    1+c0*c2    c1*c2;
      c3 c0*c1*c2*c3 1+c1*c3];
M = 256;

f = 3.6;
d_0 = 0.5;

alpha = 0.2;
beta = 0.2;

I = imread("Images/article/55.png");
W = imread("Images/article/C.png");
Mask = imread("Images/test_images/B.png");

[a,b,c] = size(I);


numZeros = sum(sum(Mask(:,:,1)==0))
