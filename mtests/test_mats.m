
pars = [0.78490597 0.7136921  0.30879194 0.32900363];
c = [0.53665215];
d = [0.48877713];
r = [0.1742107];

A = [28.637302    0           0           0
     0.7520833    0.65534526  0           0
    -1.4405712   -0.81310695  2.0540895   0
    -0.76813376   0.38873562  0.18408561  1.0502638 ];
for i=[1:length(A)]
    for j=[1:i]
        A(j,i) = A(i,j);
    end
end
disp(A);

An = [  32.67441    0           0           0 
        0.7520833   0.65534526  0           0
        1.2194667   0.70303124 -0.0514846   0
        0.45718572 -0.19220628 11.556511    0.8611462 ];
for i=[1:length(An)]
    for j=[1:i]
        An(j,i) = An(i,j);
    end
end
disp(An);

D = zeros(size(An));
L = zeros(size(An));
for i =[1:length(An)]
    D(i,i) = An(i,i);
    L(i,i) = 1.0;
end

for i=[1:length(An)]
    for j=[1:i-1]
        L(i,j)=An(i,j);
    end
end
disp(D);
disp(L);
disp(L*D*L.');

disp(A \ g');
