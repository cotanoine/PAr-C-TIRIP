function I_filter = gfilter(I,M)

l = size(I,1);
h = size(I,2);

I_filter = I;

for i=2:l-1
    for j=2:h-1
        I_filter(i,j,:) = [0,0,0];
        for k=1:3
            for l=1:3
                I_filter(i,j,:) = I_filter(i,j,:) + I(i+(2-k),j+(2-l),:)*M(k,l);
            end
        end
    end
end

end

