function rhs=nls_rhs(t,ut,dummy,k)
u = ifft(ut);
rhs = -(i/2)*(k.^2).*ut + i*fft((abs(u).^2).*u);
