function [J] = apply_haar2(I, W, param, A, M)

[Ia,Ih,Iv,Id] = haart2(I,1);
[Wa,Wh,Wv,Wd] = swt2(W,1,'haar');


l = size(Ia,1);
h = size(Ia,2);
Wa = resize(Wa, [l h]);
Wh = resize(Wh, [l h]);
Wv = resize(Wv, [l h]);
Wd = resize(Wd, [l h]);

Wa_prime = Wa;
Wh_prime = Wh;
Wv_prime = Wv;
Wd_prime = Wd;



for x=0:l-1
    for y=0:h-1
        for k=1:3
            v = Wa(x+1,y+1,k);
            xp = mod(A(1,1)*x + A(1,2)*y + A(1,3)*v,M);
            yp = mod(A(2,1)*x + A(2,2)*y + A(2,3)*v,M);
            vp = mod(A(3,1)*x + A(3,2)*y + A(3,3)*v,M);
            Wa_prime(xp+1,yp+1,k) = vp;

            v = Wh(x+1,y+1,k);
            xp = mod(A(1,1)*x + A(1,2)*y + A(1,3)*v,M);
            yp = mod(A(2,1)*x + A(2,2)*y + A(2,3)*v,M);
            vp = mod(A(3,1)*x + A(3,2)*y + A(3,3)*v,M);
            Wh_prime(xp+1,yp+1,k) = vp;

            v = Wv(x+1,y+1,k);
            xp = mod(A(1,1)*x + A(1,2)*y + A(1,3)*v,M);
            yp = mod(A(2,1)*x + A(2,2)*y + A(2,3)*v,M);
            vp = mod(A(3,1)*x + A(3,2)*y + A(3,3)*v,M);
            Wv_prime(xp+1,yp+1,k) = vp;

            v = Wa(x+1,y+1,k);
            xp = mod(A(1,1)*x + A(1,2)*y + A(1,3)*v,M);
            yp = mod(A(2,1)*x + A(2,2)*y + A(2,3)*v,M);
            vp = mod(A(3,1)*x + A(3,2)*y + A(3,3)*v,M);
            Wd_prime(xp+1,yp+1,k) = vp;
        end
    end
end

Ja = Ia + param*Wa_prime;
Jh = Ih + param*Wh_prime;
Jv = Iv + param*Wv_prime;
Jd = Id + param*Wd_prime;

l = size(I,1);
h = size(I,2);

J = imresize(iswt2(Ja,Jh,Jv,Jd,'haar'), [l h]);

end