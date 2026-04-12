function I_tilde = blind(I)

omega = [0.4,0.35,0.25];

I_tilde = I;

for k=1:3
    maxk = max(max(I(:,:,k)));
    muk = mean(mean(I(:,:,k)));
    I_tilde(:,:,k) = I_tilde(:,:,k) - omega(k)*( maxk - muk)/2;
end

H4 = ones(3,3)/9;
I_tilde = gfilter(I_tilde,H4);

end

