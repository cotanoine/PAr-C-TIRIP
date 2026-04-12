function [J] = apply_haar(I, W, param)

[Ia,Ih,Iv,Id] = haart2(I,1);
[Wa,Wh,Wv,Wd] = haart2(W,1);

Ja = Ia + param.*Wa;
Jh = Ih + param.*Wh;
Jv = Iv + param.*Wv;
Jd = Id + param.*Wd;

J = ihaart2(Ja,Jh,Jv,Jd);

%J = I+param*W;

end
