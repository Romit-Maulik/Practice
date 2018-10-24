function rhs=burgers_rhs(t,ut,dummy,k,nu)
u = ifft(ut);
uxt = (i).*(k).*ut;
ux = ifft(uxt);
uux = fft(u.*ux);
rhs = -(nu).*(k.^2).*ut - uux;




