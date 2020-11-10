clear,clc
r_p = 0:1:300;
r_d = 0:0.1:15;
f_p = zeros(1, size(r_p,2));
f_d = zeros(1, size(r_d,2));

parfor i = 1 : numel(r_d)
    i
    % view kernel
    gamma = 10;
    dfun = @(x,y,z) max(0, (1-0.5*(x.*x+y.*y+z.*z))).^gamma .* exp(-1i.*r_d(i)/sqrt(3)*(x+y+z));
    f_d(i) = integral3(dfun, -2, 2, -2, 2, -2, 2);
end

parfor i = 1 : numel(r_p)
    i
    % positional kernel
    gamma = 0.008;
    pfun = @(x,y,z) exp(-sqrt(x.*x+y.*y+z.*z)/gamma) .* exp(-1i.*r_p(i)/sqrt(3)*(x+y+z));
    f_p(i) = integral3(pfun, -2, 2, -2, 2, -2, 2);
end

f_p = real(f_p) / max(real(f_p));
f_d = real(f_d) / max(real(f_d));
f_p = max(0, f_p);
f_d = max(0, f_d);

figure;
plot(r_p, f_p);
title('positional spectrum');
figure;
plot(r_d, f_d);
title('directional spectrum');
