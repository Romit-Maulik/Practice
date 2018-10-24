function rhs=nls_rom_rhs(t,a,dummy,phi,phixx)

rhs = (i/2)*(phi.')*phixx*a + i*(phi.')*( (abs(phi*a).^2).*(phi*a));