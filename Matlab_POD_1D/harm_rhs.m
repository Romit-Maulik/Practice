function rhs=harm_rhs(t,ut,dummy,k,V)
u = ifft(ut);
rhs = (-i/2)*(k.^2).*ut - (i/2)*fft(V.*u);