function [I] = fusion(F,B,M)

l = size(F,1);
h = size(F,2);

I = B;

for i=1:l
    for j=1:h
        
        if M(i,j,1) == 0
            I(i,j,:) = B(i,j,:);
        else
            I(i,j,:) = F(i,j,:);
        end

    end
end

end

