function [J] = apply_haar(I, W, param)

[Ia,Ih,Iv,Id] = haart2(I);
[Wa,~,~,~] = haart2(W);

Ja = Ia + param*Wa;

J = ihaart2(Ja,Ih,Iv,Id);

end

