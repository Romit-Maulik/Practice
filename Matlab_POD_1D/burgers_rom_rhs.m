function rhs=burgers_rom_rhs(t,a,dummy,phi,phix,phixx)

rhs = (phi.')*phixx*a - (phi.')*( ((phi*a)).*(phix*a));