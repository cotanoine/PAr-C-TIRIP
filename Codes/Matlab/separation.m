function [F,B] = separation(I,M)

l = size(I,1);
h = size(I,2);

F = zeros(l,h,3);
B = zeros(l,h,3);

for i=1:l
    for j=1:h
        
        if M(i,j,1) == 0
            B(i,j,:) = I(i,j,:);
        else
            F(i,j,:) = I(i,j,:);
        end

    end
end

end

