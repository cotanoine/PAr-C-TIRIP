clear
addpath("Codes/Matlab");

c0=0;c1=0;c2=0;c3=0; %
A = [ 1      c0          c1;
      c2    1+c0*c2    c1*c2;
      c3 c0*c1*c2*c3 1+c1*c3];
M = 256; %

f = 3.6;
d_0 = 0.5;

alpha = 0.2;
beta = 0.2;


W = imread("Images/article/w.png");

RES = zeros(2*5,8);


for i=1:5

   
    I = imread(compose("Images/article/%d.png",i));
    Mask = imread(compose("Images/article/m%d.png",i));
    I = imresize(I, [512 512]);

    
 
    [F_I,B_I] = separation(I,Mask);
    [F_W,B_W] = separation(W,Mask);
    
    F_W_arnold = arnold(F_W,A,M);
    B_W_arnold = arnold(B_W,A,M);
    
    F_W_logistic = logistic_map(F_W_arnold,Mask,d_0,f,false);
    B_W_logistic = logistic_map(B_W_arnold,1-Mask,d_0,f,false);
    
    F_final = apply_haar(F_I, F_W_logistic, alpha);
    B_final = apply_haar(B_I, B_W_logistic, beta);
    
    I_prime = fusion(F_final,B_final,Mask);
    
    % -----------------------------------

    [I_mean,I_median,I_noise,I_rotate,I_crop,I_jpeg,I_shear] = attacks(I_prime);
    I_att = [I_prime,I_mean,I_median,I_noise,I_rotate,I_crop,I_jpeg,I_shear];


    for j=1:8

        I_prime_att = I_att(:,(1+(j-1)*512):(j*512),:);

        for k=1:2

            if k==1
                F_I_tilde = F_I;
                B_I_tilde = B_I;
            else
                I_tilde = blind(I_prime_att);
                [F_I_tilde,B_I_tilde] = separation(I_tilde,Mask);
            end

            % -----------------------------------
        

            [F_Iw,B_Iw] = separation(I_prime_att,Mask);

            
            % -----------------------------------
            
            F_W_logistic_recover = reverse_haar(F_I_tilde, F_Iw, alpha);
            B_W_logistic_recover = reverse_haar(B_I_tilde, B_Iw, beta);
            
            F_W_arnold_recover = logistic_map(F_W_logistic_recover,Mask,d_0,f,true);
            B_W_arnold_recover = logistic_map(B_W_logistic_recover,1-Mask,d_0,f,true);
            
            F_W_recover = arnold(F_W_arnold_recover,inv(A),M);
            B_W_recover = arnold(B_W_arnold_recover,inv(A),M);
            
            W_prime = fusion(F_W_recover,B_W_recover,Mask);
    
            assim = max(ssim(uint8(W_prime),uint8(W)),0);
    
            RES(2*(i-1)+k,j) = assim

        end

    end

end

RES

