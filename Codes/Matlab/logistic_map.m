function [I_logistic] = logistic_map(I,Mask,d_0,f,reverse)

l = size(I,1);
h = size(I,2);

numZeros = sum(sum(Mask(:,:,1)==0));

n = l*h-numZeros;

d = zeros(1,n);
d(1) = d_0;
for i=1:n-1
    d(i+1) = f * d(i) * (1-d(i));
end

[~,p] = sort(d);

if reverse
    [~,p] = sort(p);
end

I_logistic_flat = zeros(n,3);

indice = 1;
for i=1:l
    for j=1:h
        if Mask(i,j,1)==1
            I_logistic_flat(indice,:) = I(i,j,:);
            indice = indice+1;
        end
    end
end


I_logisitic_flat_order = I_logistic_flat(p,:);

I_logistic = zeros(l,h,3);

indice = 1;
for i=1:l
    for j=1:h
        if Mask(i,j,1)==1
            I_logistic(i,j,:) = I_logisitic_flat_order(indice,:)';
            indice = indice+1;
        end
    end
end

end

