function [E] = reverse_haar2(I, J, param, A, M)

W = (J-I)/param;
E = W;

Ai = inv(A);
l = size(I,1);
h = size(I,2);

for x=0:l-1
    for y=0:h-1
        for k=1:3
            v = W(x+1,y+1,k);
            xp = mod(Ai(1,1)*x + Ai(1,2)*y + Ai(1,3)*v,M);
            yp = mod(Ai(2,1)*x + Ai(2,2)*y + Ai(2,3)*v,M);
            vp = mod(Ai(3,1)*x + Ai(3,2)*y + Ai(3,3)*v,M);
            E(xp+1,yp+1,k) = vp;
        end
    end
end




end

