function [I_Arnold] = arnold(I,A,M)

l = size(I,1);
h = size(I,2);


I_Arnold = I;

for i=1:l
    for j=1:h
        
        I_Arnold(i,j,:) = mod(A*double(reshape(I(i,j,:),[],1)),M);

    end
end
    
end

