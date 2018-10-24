function rhs=harm_rom_rhs(t,a,dummy,phi,V,phixx)

rhs = (i/2)*(phi.')*phixx*a - (1i)*(phi.')*( V.*(phi*a));