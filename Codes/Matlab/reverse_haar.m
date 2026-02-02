function [E] = reverse_haar(I, J, W, param)

[Ja,~,~,~] = haart2(J);
[Ia,~,~,~] = haart2(I);
[~,Wh,Wv,Wd] = haart2(W);

Ea = (Ja - Ia)/param;

E = uint8(ihaart2(Ea,Wh,Wv,Wd));

end

