!---------------------------------------------------------------------------!
!   	 NS2D: two-dimensional Navier-Stokes solver in periodic domain
!---------------------------------------------------------------------------!
!   	 Domain: (x,y) in periodic [0,2pi] 
!-----------------------------------------------------------------!
! 		 two-dimensional navier-stokes solver  
!        vorticity-stream function formulation
!		 4th-order Arakawa scheme (or compact scheme) for nonlinear term
!	     4th order compact Pade scheme for spatial derivaties (linear terms)
!		 3rd order Runge-Kutta for temporal discritization
!        periodic boundary conditions only
!        
!
!		 needs:   input.txt   ==> header file (options for the solver)
!          and/or initial.dat ==> initial condition for vorticity field
!          and/or final.dat   ==> saved condition
!
!        results: fort.500	==> vorticity field (fort.200 is normalized)
!				  fort.600	==> energy spectrum (all frequencies)
!				  fort.700  ==> energy spectrum (angle avaraged)
!				  fort.300	==> mid line vorticity values (along y)
!				  fort.400	==> mid line vorticity values (along x)  
!				  fort.800	==> 2nd order structure function
!                 fort.100000 ==> filtered dns data
!                 
!
!---------------------------------------------------------------------------!
!Case: Decaying turbulence problem
!---------------------------------------------------------------------------!
!Omer San, Oklahoma State University, cfdlab.osu@gmail.com 
!Updated: June 07, 2016
!---------------------------------------------------------------------------!

program ns2d
implicit none
integer::nd,nt,nf,ion,ifil,ich,isc,isgs,idyn,ivis,NA,ipr
integer::nx,ny
integer::fw,ifile,ikolmog,idt,isolver
integer::i,j,n
double precision::re,dt,kappa2,sma,csd,delta,del,betaAD,afil
double precision::pi,lx,ly,dx,dy,rate,en0,enf0,Afor,kfor,nu
double precision::time,cpuinit,cpufinal,cfl,en,ew,cpuh,delc
double precision::Akolmog,kkolmog,cflc,dta,dtv,neuc,lambda,Rec
double precision,dimension(:,:),allocatable::w,wf

common /solver/ isolver
common /forcing/ Akolmog,kkolmog,ikolmog
common /kolmog/ Afor,kfor,nu
common /LESmodels/ isgs
common /filtering/ ifil
common /dyncoeff/ csd
common /ADmodel/ betaAD,NA
common /smaconstant/ sma,del
common /clarkconstant/ delc
common /dynconstant/ kappa2
common /Padefilter/ afil
common /viskernel/ ivis
common /dynmodel/ idyn
common /adaptive/ cflc,dta,dtv

! read input file:
open(7,file='input.txt')
read(7,*)nd
read(7,*)nt
read(7,*)re
read(7,*)dt
read(7,*)idt
read(7,*)cflc
read(7,*)neuc
read(7,*)nf
read(7,*)isolver
read(7,*)ipr
read(7,*)kfor
read(7,*)Afor
read(7,*)lambda
read(7,*)ion
read(7,*)ifil
read(7,*)afil
read(7,*)isgs
read(7,*)idyn
read(7,*)ivis
read(7,*)kappa2
read(7,*)sma
read(7,*)delta
read(7,*)NA
read(7,*)betaAD
read(7,*)isc
read(7,*)ich
close(7)
if (ich.ne.19) then
write(*,*)'check input.txt file..'
stop
end if

if (ivis.eq.2) then !correction to kappa for leith
  kappa2 = dsqrt(kappa2)**3
end if

if(ipr.eq.5) then !if Kolmogorov flow (forcing)
Rec = dsqrt(2.0d0)
ikolmog = 1
Akolmog = Afor
kkolmog = kfor
nu = dsqrt(Afor/(lambda*Rec*(kfor**4)))
re = 1.0d0/nu  !recompute Re
else
ikolmog = 0
Akolmog = 0.0d0
kkolmog = 0.0d0
nu = 1.0d0/re
end if

! some parameters
nx=nd
ny=nd

pi=4.0d0*datan(1.0d0)
lx=2.0d0*pi
ly=2.0d0*pi


dx=lx/dfloat(nx)
dy=ly/dfloat(ny)

dtv = neuc*nu/(dx**2)

!filter length scale
del=dsqrt(dx*delta*dy*delta)
delc=del

ifile=0
time=0.0d0
csd = 0.0d0


! allocate the vorticity array:
allocate(w(-2:nx+2,-2:ny+2))
allocate(wf(-2:nx+2,-2:ny+2))

! initial conditions:
call ic(nx,ny,dx,dy,ion,ipr,time,w)

!open(77,file='spec-time.plt')
!write(77,*)'variables ="k","E(k)"'


! open(5,file='initial.plt')
! write(5,*) 'variables ="x","y","w"'
! write(5,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',time,'"'
! do j=0,ny
! do i=0,nx
! write(5,52) dfloat(i)*dx,dfloat(j)*dy,w(i,j)
! end do
! end do
! close(5)



! ! write initial voryicity field in tecplot format:
call outw(ifile,nx,ny,dx,dy,time,w)
! call spec(ifile,nx,ny,time,w)
! call line(ifile,nx,ny,dx,dy,time,w)
! call strfun(ifile,nx,ny,time,w)


! compute history:
call history(nx,ny,dx,dy,dt,w,cfl,en,ew)
en0=en

!adaptive time stepping
if(idt.eq.1) then
dt = dta
end if


!check for initial cfl number
111 continue
if (cfl.ge.cflc) then
dt = 0.5d0*dt
cfl= 0.5d0*cfl
nt = nt*2
goto 111
end if

open(66,file="cfl.txt")
write(66,*)'idt= ',idt
write(66,*)'time step= ',dt
write(66,*)'dtv = ',dtv
write(66,*)'CFL = ',cfl
write(66,*)'nu = ',nu
write(66,*)'1/nu = ',1.0d0/nu
close(66)


open(11,file='history_rate.plt')
write(11,*) 'variables ="t","e"'

open(12,file='history_rate_filtered.plt')
write(12,*) 'variables ="t","e"'
    

! write historty file:
open(9,file='history.plt')
write(9,*) 'variables ="t","E","Q","cfl","max_w","min_w"'
write(9,19) time,en,ew,cfl,maxval(w),minval(w)
if(isc.eq.1) write(*,18) time,en,cfl

! write historty file for dynamic coefficient:
if(isgs.eq.9) then
	if (ivis.eq.1) then
	open(99,file='history_dyn.plt')
	write(99,*) 'variables ="t","c"'
    else
	open(99,file='history_dyn.plt')
	write(99,*) 'variables ="t","c"'
    end if
end if


!Filtered DNS data
if (isgs.eq.0) then
! compute filtered history
call filter(nx,ny,w,wf)
call history(nx,ny,dx,dy,dt,wf,cfl,en,ew)
enf0=en

! write filtered historty file
open(8,file='history_filtered.plt')
write(8,*) 'variables ="t","E","Q","cfl","max_w","min_w"'
write(8,19) time,en,ew,cfl,maxval(wf),minval(wf)

	! call outw(100000+ifile,nx,ny,dx,dy,time,wf)
	! call spec(100000+ifile,nx,ny,time,wf)
 !    call line(100000+ifile,nx,ny,dx,dy,time,wf)
	! call strfun(100000+ifile,nx,ny,time,wf)

end if

call cpu_time(cpuinit)

if (nt.le.nf) nf=nt
fw= max(1,nt/nf) !writing frequency


!=====================================
!time integration is starting...
!=====================================
do n=1,nt

	time=time+dt

	!tvd 3rd-order rk scheme (3-stage)
	call rk3tvd(nx,ny,dx,dy,dt,re,w)

	
	! write on the log file
	call history(nx,ny,dx,dy,dt,w,cfl,en,ew)
	write(9,19) time,en,ew,cfl,maxval(w),minval(w)
    
	if(isc.eq.1) write(*,18) time,en,cfl
	
	if(isgs.eq.9) then
		if (ivis.eq.1) then
		write(99,*) time,dsqrt(csd/(dx*dy))
    	else
		write(99,*) time,(csd/(del**3))**(1.0d0/3.0d0)
    	end if
	end if

	!adaptive time stepping
	if(idt.eq.1) then
	dt = dta
	end if

	!compute dissipation rate
    rate = -(en-en0)/dt
    en0=en
    write(11,*)time-dt*0.5d0, rate
    
	if (isgs.eq.0) then	
	call filter(nx,ny,w,wf)
	call history(nx,ny,dx,dy,dt,wf,cfl,en,ew)
	write(8,19) time,en,ew,cfl,maxval(wf),minval(wf)
    !compute dissipation rate
    rate = -(en-enf0)/dt
    enf0=en
    write(12,*)time-dt*0.5d0, rate
	end if



	!writing vorticity field:
	if (mod(n,fw).eq.0) then
	ifile=ifile+1

	call outw(ifile,nx,ny,dx,dy,time,w)
	! call spec(ifile,nx,ny,time,w)
    ! call line(ifile,nx,ny,dx,dy,time,w)
	! call strfun(ifile,nx,ny,time,w)	
	
	! if (isgs.eq.0) then	
	! call filter(nx,ny,w,wf)
	! call outw(100000+ifile,nx,ny,dx,dy,time,wf)
	! call spec(100000+ifile,nx,ny,time,wf)
 !    call line(100000+ifile,nx,ny,dx,dy,time,wf)
	! call strfun(100000+ifile,nx,ny,time,wf)
	! end if
		
	end if
	

	call cpu_time(cpufinal)
	cpuh=(cpufinal-cpuinit)/60./60.
	if(cpuh.ge.503.0d0) goto 44

end do

44 continue

if (isgs.eq.0) close(8)
if (isgs.eq.9) close(99)
close(9)
close(11)
close(12)
!close(77)

call cpu_time(cpufinal)

open(66,file="cpu.txt")
write(66,*)'cpu time = ',(cpufinal-cpuinit), '  second'
write(66,*)'cpu time = ',(cpufinal-cpuinit)/60./60., '  hrs'
write(66,*)'Max time = ',time
write(66,*)'Number of time = ',n
write(66,*)'Time step = ',dt
close(66)

call saved(nx,ny,time,w)


! open(5,file='final.plt')
! write(5,*) 'variables ="x","y","w"'
! write(5,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',time,'"'
! do j=0,ny
! do i=0,nx
! write(5,52) dfloat(i)*dx,dfloat(j)*dy,w(i,j)
! end do
! end do
! close(5)

51 format(a16,i8,a4,i8,a10,f10.4,a3)
52 format(3es16.6)
19 format(6es16.6)
18 format(3es18.8)


!Exact solution for TGV problem
if (ipr.eq.1) call tgvnorm(nx,ny,dx,dy,time,re,w)


end 

!---------------------------------------------------------------------------!
!Compute 2D vorticity field from the energy spectrum
!Periodic, equidistant grid
!---------------------------------------------------------------------------!
subroutine spec2field2d(nx,ny,dx,dy,w)
implicit none
integer::nx,ny,ni,nj,ii,jj
double precision ::w(-2:nx+2,-2:ny+2)
double precision ::ran,pi,kk,E4,dx,dy
double precision,parameter:: tiny=1.0d-10
double precision,allocatable ::data1d(:),phase2d(:,:,:),ksi(:,:),eta(:,:)
double precision,allocatable ::kx(:),ky(:),ww(:,:)
integer::i,j,k,isign,ndim,nn(2),seed

seed = 19

!expand it to dns grid

ni = nx
nj = ny

nx = 2048
ny = 2048

if (ni.ge.2048) nx=ni
if (nj.ge.2048) ny=nj

ii = nx/ni
jj = ny/nj

ndim =2
nn(1)=nx
nn(2)=ny

allocate(kx(0:nx-1),ky(0:ny-1))
allocate(ksi(0:nx/2,0:ny/2),eta(0:nx/2,0:ny/2))
allocate(data1d(2*nx*ny))
allocate(phase2d(2,0:nx-1,0:ny-1))
allocate(ww(0:nx,0:ny))

!Set seed for the random number generator between [0,1]
CALL RANDOM_SEED(seed)

pi = 4.0d0*datan(1.0d0)


!Wave numbers 
do i=0,nx/2-1
kx(i)      = dfloat(i)
kx(i+nx/2) = dfloat(i-nx/2)
end do
kx(0) = tiny

do j=0,ny/2-1
ky(j)      = dfloat(j)
ky(j+ny/2) = dfloat(j-ny/2)
end do
ky(0) = tiny

!Random numbers in the first quadrant
do j=0,ny/2
do i=0,nx/2
CALL RANDOM_NUMBER(ran)
ksi(i,j) =2.0d0*pi*ran
end do
end do

do j=0,ny/2
do i=0,nx/2
CALL RANDOM_NUMBER(ran)
eta(i,j) =2.0d0*pi*ran
end do
end do

!Random phase
do j=0,ny-1
do i=0,nx-1
phase2d(1,i,j)       = 0.0d0
phase2d(2,i,j)       = 0.0d0
end do
end do
  
do j=1,ny/2-1
do i=1,nx/2-1
!I.st
phase2d(1,i,j)       = dcos(ksi(i,j)+eta(i,j)) 
phase2d(2,i,j)       = dsin(ksi(i,j)+eta(i,j)) 
!II.nd
phase2d(1,nx-i,j)    = dcos(-ksi(i,j)+eta(i,j)) 
phase2d(2,nx-i,j)    = dsin(-ksi(i,j)+eta(i,j)) 
!IV.th
phase2d(1,i,ny-j)    = dcos(ksi(i,j)-eta(i,j)) 
phase2d(2,i,ny-j)    = dsin(ksi(i,j)-eta(i,j)) 
!III.rd
phase2d(1,nx-i,ny-j) = dcos(-ksi(i,j)-eta(i,j)) 
phase2d(2,nx-i,ny-j) = dsin(-ksi(i,j)-eta(i,j)) 
end do
end do


!vorticity amplitudes in Fourier space 
k=1
do j=0,ny-1
do i=0,nx-1   
    kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
    data1d(k)   =  dsqrt(kk*E4(kk)/pi)*phase2d(1,i,j)
	data1d(k+1) =  dsqrt(kk*E4(kk)/pi)*phase2d(2,i,j)   
k = k + 2
end do
end do

!find the velocity in physical space
!forward fourier transform
isign= 1
call fourn(data1d,nn,ndim,isign)


k=1
do j=0,ny-1
do i=0,nx-1
ww(i,j)=data1d(k)
k=k+2
end do
end do


! periodicity
do j=0,ny-1
ww(nx,j)=ww(0,j)
end do
do i=0,nx
ww(i,ny)=ww(i,0)
end do

!back to the local grid
nx = ni
ny = nj
do j=0,ny
do i=0,nx
w(i,j)=ww(i*ii,j*jj)
end do
end do

open(5,file='initial-w.plt')
write(5,*) 'variables ="x","y","w"'
write(5,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',0.0,'"'
do j=0,ny
do i=0,nx
write(5,52) dfloat(i)*dx,dfloat(j)*dy,w(i,j)
end do
end do
close(5)


deallocate(data1d,phase2d,ksi,eta,ww)

!write input files:
!write files when dns starts
if(nx.eq.2048) then
open(3,file='i2048.dat')
write(3,*) nx,ny
write(3,*)((w(i,j), i=0,nx), j=0,ny)
close(3)

open(3,file='i1024.dat')
write(3,*) nx/2,ny/2
write(3,*)((w(i*2,j*2), i=0,nx/2), j=0,ny/2)
close(3)

open(3,file='i512.dat')
write(3,*) nx/4,ny/4
write(3,*)((w(i*4,j*4), i=0,nx/4), j=0,ny/4)
close(3)

open(3,file='i256.dat')
write(3,*) nx/8,ny/8
write(3,*)((w(i*8,j*8), i=0,nx/8), j=0,ny/8)
close(3)

open(3,file='i128.dat')
write(3,*) nx/16,ny/16
write(3,*)((w(i*16,j*16), i=0,nx/16), j=0,ny/16)
close(3)

open(3,file='i64.dat')
write(3,*) nx/32,ny/32
write(3,*)((w(i*32,j*32), i=0,nx/32), j=0,ny/32)
close(3)
end if

51 format(a16,i8,a4,i8,a10,f10.4,a3)
52 format(3es16.6)
return
end


!---------------------------------------------------------------------------!
!Given energy spectrum
!---------------------------------------------------------------------------!
double precision function E4(kr)
implicit none
double precision:: kr,pi,c,k0
k0 = 10.0d0
pi = 4.0d0*datan(1.0d0)
c = 4.0d0/(3.0d0*dsqrt(pi)*(k0**5))
!c = 1.0d0/(4.0d0*pi*(k0**6))
!c = 1.0d0/(2.0d0*pi*(k0**6))
E4 = c*(kr**4)*dexp(-(kr/k0)**2)
end


!-------------------------------------------------------------------------!
!initial conditions
!-------------------------------------------------------------------------!
subroutine tgvnorm(nx,ny,dx,dy,time,re,w)
implicit none
integer :: nx,ny,i,j,nq
double precision::time,re,xx,yy,sum,dx,dy
double precision, dimension (-2:nx+2,-2:ny+2)  :: w
double precision,dimension(:,:),allocatable:: we

! exact solution for tgv
allocate(we(-2:nx+2,-2:ny+2))
nq = 4 !number of arrays
do j=0,ny
do i=0,nx
    xx=dfloat(i)*dx
	yy=dfloat(j)*dy
we(i,j) = 2.0d0*dfloat(nq) &
        *dcos(dfloat(nq)*xx) &
        *dcos(dfloat(nq)*yy) &
        *dexp(-2.0d0*dfloat(nq)*dfloat(nq)*time/re)
end do
end do

! l2 avaraged norm
sum = 0.0d0
do j=0,ny
do i=0,nx
sum = sum + (w(i,j)-we(i,j))**2
end do
end do
sum = dsqrt(sum/(dfloat(nx+1)*dfloat(ny+1)))

open(17,file='tgv.txt')
write(17,*)"Time: ", time
write(17,*)"Resolution: ", nx
write(17,*)"L2 norm: ", sum
close(17)

return
end


!-------------------------------------------------------------------------!
!initial conditions
!-------------------------------------------------------------------------!
subroutine ic(nx,ny,dx,dy,ion,ipr,time,w)
implicit none
integer :: nx,ny,nxi,nyi,i,j,ion,ipr,nq
double precision::time,dx,dy,pi,sigma,xc1,xc2,yc1,yc2,xx,yy,delta
double precision::Afor,kfor,nu,noiseamp
double precision, dimension (-2:nx+2,-2:ny+2)  :: w

common /kolmog/ Afor,kfor,nu


pi =4.0d0*datan(1.0d0)

if (ipr.eq.1) then
!Taylor Green Vortex
    nq = 4 !number of arrays
    
	do j=0,ny
	do i=0,nx
    xx=dfloat(i)*dx
	yy=dfloat(j)*dy
    
	w(i,j) =  2.0d0*dfloat(nq)*dcos(dfloat(nq)*xx) &
							  *dcos(dfloat(nq)*yy)
	end do
	end do
      
else if (ipr.eq.2) then
!Two-gaussian vortex (vortex merging)
	sigma=1.0d0*pi
	xc1 =pi-pi/4.0d0
	yc1 =pi
	xc2 =pi+pi/4.0d0
	yc2 =pi

	do j=0,ny
	do i=0,nx
	
	xx=dfloat(i)*dx
	yy=dfloat(j)*dy
	
	w(i,j) =  dexp(-sigma*((xx-xc1)**2 + (yy-yc1)**2)) &
	         +dexp(-sigma*((xx-xc2)**2 + (yy-yc2)**2)) 
	end do
	end do
    
else if (ipr.eq.3) then
!Double shear layer problem

	delta = 0.05d0				!perturbation amplitude
	sigma = 30.0d0/(2.0d0*pi)	!thickness parameter

	do j=0,ny
	do i=0,nx

	xx=dfloat(i)*dx
	yy=dfloat(j)*dy
	
	if (yy.le.pi) then
	w(i,j) = delta*dcos(xx) - sigma/(dcosh(sigma*(yy-pi/2.0d0)))**2
	else
	w(i,j) = delta*dcos(xx) + sigma/(dcosh(sigma*(3.0d0*pi/2.0d0-yy)))**2
	end if
	
	end do
	end do
    
else if (ipr.eq.4) then
  
if (ion.eq.1) then 
	open(3,file='i2048.dat')
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check initial file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)

else if (ion.eq.2) then
	open(3,file='i1024.dat')
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check initial file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)
    
else if (ion.eq.3) then

	open(3,file='i512.dat')
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check initial file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)

else if (ion.eq.4) then

	open(3,file='i256.dat')
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check initial file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)


else if (ion.eq.5) then

	open(3,file='i128.dat')
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check initial file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)


else if (ion.eq.6) then

	open(3,file='i64.dat')
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check initial file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)

else

	call spec2field2d(nx,ny,dx,dy,w)
  
end if

else if (ipr.eq.5) then !Kolmogorov flow

!add noise to trigger instability
	noiseamp=0.05d0
	do j=0,ny
	do i=0,nx
	xx=dfloat(i)*dx
	yy=dfloat(j)*dy	
	w(i,j) = (Afor/(nu*kfor**2))*(dsin(kfor*yy) + noiseamp*dsin(kfor*xx)*dsin(kfor*yy))	
	end do
	end do
  
else
! from 'final.dat' file
	open(3,file='final.dat')
	read(3,*)time
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check final.dat file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)

end if


!extend for periodic b.c.
call bc(nx,ny,w)

return
end


!-------------------------------------------------------------------------!
!periodic boundary conditions
!-------------------------------------------------------------------------!
subroutine bc(nx,ny,u)
implicit none
integer :: nx,ny,i,j
double precision, dimension (-2:nx+2,-2:ny+2)  :: u

!extend for periodic b.c.
do i=0,nx
u(i,-2)  = u(i,ny-2)
u(i,-1)  = u(i,ny-1)
u(i,ny+1)= u(i,1)
u(i,ny+2)= u(i,2)
end do
do j=-2,ny+2
u(-2,j)  = u(nx-2,j)
u(-1,j)  = u(nx-1,j)
u(nx+1,j)= u(1,j)
u(nx+2,j)= u(2,j)
end do

return
end



!------------------------------------------------------------------!
!Approximate deconvolution method for 2D data
!
!NA: order of van Cittert approximation
!
!compute unfiltered quantity u from the filtered variable uf
!by repeated filtering (van Cittert series)
!also known iterative inverse filtering
!
!filtering operation in physical space by discrete filters
!------------------------------------------------------------------!
subroutine adm(nx,ny,uf,u)
implicit none
integer::nx,ny,NA,i,j,k
double precision::betaAD
double precision,dimension(-2:nx+2,-2:ny+2):: u,uf
double precision, dimension (:,:), allocatable :: ug

common /ADModel/ betaAD,NA

allocate(ug(-2:nx+2,-2:ny+2))

!initial guess
!k=0 
do j=0,ny
do i=0,nx
u(i,j) = uf(i,j)
end do
end do

!k>0    
do k = 1, NA    
    !compute filtered value of guess
    call filter(nx,ny,u,ug)  
	do j=0,ny
	do i=0,nx
		u(i,j) = u(i,j) + betaAD*(uf(i,j) - ug(i,j))
	end do
    end do
end do

deallocate(ug)

!extend for periodic b.c.
call bc(nx,ny,u)

return
end


!-----------------------------------------------------------------!
!Pade Filter in 2D (fourth order)
!-----------------------------------------------------------------!
subroutine pade4filter2d(nx,ny,w,wf)
implicit none
integer::nx,ny,i,j
double precision::w(-2:nx+2,-2:ny+2),wf(-2:nx+2,-2:ny+2)
double precision, dimension (:), allocatable:: a,b
double precision, dimension(:,:),allocatable::g


allocate(g(-2:nx+2,-2:ny+2))

! filter in x direction (periodic)
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = w(i,j)
	end do
		call filterPade4p(nx,a,b)
	do i=0,nx
	g(i,j) = b(i)
	end do
end do
deallocate(a,b)


! filter in y direction (wall)
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = g(i,j)
	end do
		call filterPade4p(ny,a,b)
	do j=0,ny
	wf(i,j) = b(j)
	end do
end do
deallocate(a,b)

deallocate(g)


return
end

!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data
!Pade forth-order:  -0.5 < afil < 0.5
!Periodic
!---------------------------------------------------------------------------!
subroutine filterPade4p(n,u,uf)
implicit none
integer::n,i
double precision ::afil
double precision ::u(0:n),uf(0:n)
double precision ::alpha,beta
double precision, dimension (0:n-1):: a,b,c,r,x 

common /Padefilter/ afil

do i=0,n-1
a(i) = afil
b(i) = 1.0d0
c(i) = afil
end do

do i=2,n-2
r(i) = (0.625d0 + 0.75d0*afil)*u(i) &
      +(0.5d0 + afil)*0.5d0*(u(i-1)+u(i+1))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(i-2)+u(i+2))
end do

r(1) = (0.625d0 + 0.75d0*afil)*u(1) &
      +(0.5d0 + afil)*0.5d0*(u(0)+u(2))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(n-1)+u(3))

r(0) = (0.625d0 + 0.75d0*afil)*u(0) &
      +(0.5d0 + afil)*0.5d0*(u(n-1)+u(1))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(n-2)+u(2))

r(n-1) = (0.625d0 + 0.75d0*afil)*u(n-1) &
      +(0.5d0 + afil)*0.5d0*(u(n-2)+u(n))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(n-3)+u(1))      
      
     
alpha = afil
beta  = afil

call ctdms(a,b,c,alpha,beta,r,x,0,n-1) 

do i=0,n-1
uf(i)=x(i)
end do
uf(n)=uf(0)

return 
end




!-----------------------------------------------------------------!
!Filter
!-----------------------------------------------------------------!
subroutine filter(nx,ny,w,wf)
implicit none
integer ::nx,ny,ifil,i,j
double precision::w(-2:nx+2,-2:ny+2),wf(-2:nx+2,-2:ny+2),dd

common /filtering/ ifil

call bc(nx,ny,w)

if (ifil.eq.1) then !trapezoidal filter
dd=1.0d0/16.0d0

do j=0,ny
do i=0,nx
wf(i,j) = dd*(4.0d0*w(i,j) &
       + 2.0d0*(w(i+1,j) + w(i-1,j) + w(i,j+1) + w(i,j-1)) &
	   + w(i+1,j+1) + w(i-1,j-1) + w(i+1,j-1) + w(i-1,j+1))
end do
end do

else
call pade4filter2d(nx,ny,w,wf)
end if

call bc(nx,ny,wf)

return
end 



!-----------------------------------------------------------------!
! tvd 3rd-order rk scheme
!-----------------------------------------------------------------!
subroutine rk3tvd(nx,ny,dx,dy,dt,re,w)
implicit none
integer ::nx,ny,i,j,ips
double precision::dx,dy,dt,re,w(-2:nx+2,-2:ny+2),oneth,twoth
double precision, dimension (:,:), allocatable :: s,f,w1,w2

common /ellp/ ips

allocate(w1(-2:nx+2,-2:ny+2),w2(-2:nx+2,-2:ny+2))
allocate(s(-2:nx+2,-2:ny+2),f(-2:nx+2,-2:ny+2))

oneth = 1.0d0/3.0d0
twoth = 2.0d0/3.0d0

!poisson solver
call fps4(nx,ny,dx,dy,-w,s)

call rhs(nx,ny,dx,dy,re,w,s,f)

do j=0,ny
do i=0,nx
w1(i,j) = w(i,j) + dt*f(i,j)
end do
end do
call bc(nx,ny,w1)


!poisson solver
call fps4(nx,ny,dx,dy,-w1,s)

call rhs(nx,ny,dx,dy,re,w1,s,f)

do j=0,ny
do i=0,nx
w2(i,j) = 0.75d0*w(i,j) + 0.25d0*w1(i,j) + 0.25d0*dt*f(i,j)
end do
end do
call bc(nx,ny,w2)

!poisson solver
call fps4(nx,ny,dx,dy,-w2,s)

call rhs(nx,ny,dx,dy,re,w2,s,f)

do j=0,ny
do i=0,nx
w(i,j) = oneth*w(i,j) + twoth*w2(i,j) + twoth*dt*f(i,j)
end do
end do

call bc(nx,ny,w)

deallocate(s,f,w1,w2)

return
end


!-------------------------------------------------------------------------!
! evaluation of rhs terms 
! both convertive and viscous terms are treated explicitly
!-------------------------------------------------------------------------!
subroutine rhs(nx,ny,dx,dy,re,w,s,f)
implicit none
integer :: i,j,nx,ny,isgs,ikolmog
double precision :: dx,dy,re,Akolmog,kkolmog,yy
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: f
double precision, dimension (:,:), allocatable :: lp,jc,vt,r

common /LESmodels/ isgs
common /forcing/ Akolmog,kkolmog,ikolmog


call bc(nx,ny,w)
call bc(nx,ny,s)

!------------------------------!
! viscous terms:
!------------------------------!
allocate(lp(-2:nx+2,-2:ny+2))
call laplacian(nx,ny,dx,dy,w,lp)

!------------------------------!
! convective terms:
!------------------------------!
allocate(jc(-2:nx+2,-2:ny+2))
call jacobian(nx,ny,dx,dy,w,s,jc)

!------------------------------!
!compute rhs
!------------------------------!
do j=0,ny
do i=0,nx
f(i,j)=lp(i,j)/re - jc(i,j)
end do
end do

!------------------------------!
!Add forcing
!------------------------------!
if(ikolmog.eq.1) then
do j=0,ny
yy = dfloat(j)*dy
do i=0,nx
f(i,j)=f(i,j) + Akolmog*dsin(kkolmog*yy)
end do
end do
end if


!------------------------------!
!SGS terms:
!------------------------------!
if(isgs.ne.0) then
    
if(isgs.eq.1) then
  	allocate(vt(-2:nx+2,-2:ny+2))
	call smagor(nx,ny,dx,dy,s,vt)
    
    do j=0,ny
	do i=0,nx
	f(i,j)=f(i,j) + vt(i,j)*lp(i,j)
	end do
	end do
    
	deallocate(vt)

else if(isgs.eq.2) then
  	allocate(vt(-2:nx+2,-2:ny+2))
	call leith(nx,ny,dx,dy,s,vt)
    
    do j=0,ny
	do i=0,nx
	f(i,j)=f(i,j) + vt(i,j)*lp(i,j)
	end do
	end do
    
	deallocate(vt)
  
else if(isgs.eq.3) then
    allocate(r(-2:nx+2,-2:ny+2))
  	call ss(nx,ny,dx,dy,w,s,jc,r)
    do j=0,ny
	do i=0,nx
	f(i,j)=f(i,j) + r(i,j)
	end do
	end do
 	deallocate(r)  
     
else if(isgs.eq.4) then
   
    allocate(r(-2:nx+2,-2:ny+2))
  	call layton(nx,ny,jc,r)
    do j=0,ny
	do i=0,nx
	f(i,j)=f(i,j) + r(i,j)
	end do
	end do
 	deallocate(r)  
   
else if(isgs.eq.5) then
  
    allocate(r(-2:nx+2,-2:ny+2))
  	call adles(nx,ny,dx,dy,w,s,jc,r)
    do j=0,ny
	do i=0,nx
	f(i,j)=f(i,j) + r(i,j)
	end do
	end do
 	deallocate(r)  

else if(isgs.eq.6) then !not sure it is correct
  
  	allocate(r(-2:nx+2,-2:ny+2))
	call grad(nx,ny,dx,dy,s,r)
    
    do j=0,ny
	do i=0,nx
	f(i,j)=f(i,j) + r(i,j)
	end do
	end do
    
	deallocate(vt)


else if(isgs.eq.9) then 
  
    allocate(vt(-2:nx+2,-2:ny+2))
	call dyneddy(nx,ny,dx,dy,w,s,lp,jc,vt)
    
    do j=0,ny
	do i=0,nx
	f(i,j)=f(i,j) + vt(i,j)*lp(i,j)
	end do
	end do
    
	deallocate(vt)
    
	
end if

        
end if


deallocate(lp,jc)

return
end


!-------------------------------------------------------------------------!
! compute AS-LES
!-------------------------------------------------------------------------!
subroutine adles(nx,ny,dx,dy,w,s,jc,r)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy, afil, random_val
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: jc,r
double precision, dimension (:,:), allocatable :: ja,jaf
double precision, dimension (:,:), allocatable :: sa,wa

common /Padefilter/ afil

random_val = rand(1)

if (random_val > 0.5d0) then
afil = 0.1
else
afil = 0.0
end if

!AD process
allocate(sa(-2:nx+2,-2:ny+2))
allocate(wa(-2:nx+2,-2:ny+2))
	
call adm(nx,ny,s,sa)
call adm(nx,ny,w,wa)


	!compute jacobian of ad variables
	allocate(ja(-2:nx+2,-2:ny+2))
	call jacobian(nx,ny,dx,dy,wa,sa,ja)

	!compute filtered jacobian
	allocate(jaf(-2:nx+2,-2:ny+2))
	call filter(nx,ny,ja,jaf)

	!compute scale similarity
	do j=0,ny
	do i=0,nx 
		r(i,j) = jc(i,j)-jaf(i,j)
	end do
	end do

deallocate(wa,sa,ja,jaf)

return
end


!-------------------------------------------------------------------------!
! compute scale similarity
!-------------------------------------------------------------------------!
subroutine ss(nx,ny,dx,dy,w,s,jc,r)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: jc,r
double precision, dimension (:,:), allocatable :: wf,sf,fjc,jcf


	!compute jacobian of filtered variables
	allocate(wf(-2:nx+2,-2:ny+2))
	allocate(sf(-2:nx+2,-2:ny+2))
	allocate(fjc(-2:nx+2,-2:ny+2))
	call filter(nx,ny,w,wf)
	call filter(nx,ny,s,sf)

	call jacobian(nx,ny,dx,dy,wf,sf,fjc)

	!compute filtered jacobian
	allocate(jcf(-2:nx+2,-2:ny+2))
	call filter(nx,ny,jc,jcf)

	!compute scale similarity
	do j=0,ny
	do i=0,nx 
		r(i,j) = fjc(i,j)-jcf(i,j)
	end do
	end do

deallocate(wf,sf,fjc,jcf)

return
end

!-------------------------------------------------------------------------!
! compute scale similarity (layton)
!-------------------------------------------------------------------------!
subroutine layton(nx,ny,jc,r)
implicit none
integer :: i,j,nx,ny
double precision, dimension (-2:nx+2,-2:ny+2)  :: jc,r
double precision, dimension (:,:), allocatable :: jcf


	!compute filtered jacobian
	allocate(jcf(-2:nx+2,-2:ny+2))
	call filter(nx,ny,jc,jcf)

	!compute scale similarity
	do j=0,ny
	do i=0,nx 
		r(i,j) = jc(i,j)-jcf(i,j)
	end do
	end do

deallocate(jcf)


return
end

!-----------------------------------------------------------------!
!Compute clarks gradient model 
!i.e., see Legras's paper 
!-----------------------------------------------------------------!
subroutine grad(nx,ny,dx,dy,s,r)
implicit none
integer ::nx,ny,i,j
double precision::dx,dy,delc,temp
double precision::s(-2:nx+2,-2:ny+2),r(-2:nx+2,-2:ny+2)
double precision, dimension (:), allocatable   :: a,b 
double precision, dimension (:,:), allocatable :: d,e,f,g,h

common /clarkconstant/ delc


allocate(d(0:nx,0:ny))
allocate(e(0:nx,0:ny))
allocate(f(0:nx,0:ny))
allocate(g(0:nx,0:ny))
allocate(h(0:nx,0:ny))

! compute strain

! sx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = s(i,j)
	end do
		call c4dp(a,b,dx,nx)    
	do i=0,nx
	e(i,j) = b(i)
	end do
end do
deallocate(a,b)

! s_xy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = e(i,j)
	end do
		call c4dp(a,b,dy,ny)
	do j=0,ny
	f(i,j) = b(j)
	end do
end do
deallocate(a,b)


! s_xy_xx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = f(i,j)
	end do
		call c4ddp(a,b,dx,nx)
	do i=0,nx
	e(i,j) = b(i)
	end do
end do
deallocate(a,b)

! s_xy_yy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = f(i,j)
	end do
		call c4ddp(a,b,dy,ny)
	do j=0,ny
	g(i,j) = e(i,j) + b(j)
	end do
end do
deallocate(a,b)


! sxx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = s(i,j)
	end do
		call c4ddp(a,b,dx,nx)
	do i=0,nx
	e(i,j) = b(i)
	end do
end do
deallocate(a,b)

! sxx_xx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = e(i,j)
	end do
		call c4ddp(a,b,dx,nx)
	do i=0,nx
	h(i,j) = b(i)
	end do
end do
deallocate(a,b)


! syy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4ddp(a,b,dy,ny)
	do j=0,ny
	d(i,j) =  b(j)
	end do
end do
deallocate(a,b)

! syy_yy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = d(i,j)
	end do
		call c4ddp(a,b,dy,ny)
	do j=0,ny
	h(i,j) =  b(j) - h(i,j)
	end do
end do
deallocate(a,b)


temp = -delc*delc/12.0d0
!gradiant model
do j=0,ny
do i=0,nx
r(i,j) = temp*(g(i,j)*(e(i,j)-d(i,j)) + f(i,j)*h(i,j))
end do
end do


deallocate(d,e,f,g,h)


return
end 

    
!-------------------------------------------------------------------------!
! compute eddy viscosity
!-------------------------------------------------------------------------!
subroutine smagor(nx,ny,dx,dy,s,vt)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy,sma,del,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: s
double precision, dimension (-2:nx+2,-2:ny+2)  :: vt
double precision, dimension (:,:), allocatable :: st

common /smaconstant/ sma,del
common /smag/ csd

	allocate(st(-2:nx+2,-2:ny+2))
	call strain(nx,ny,dx,dy,s,st)

	csd = sma*sma*del*del

	do j=0,ny
	do i=0,nx
	vt(i,j)=csd*st(i,j)
	end do
	end do
	
	deallocate(st)

return
end

!-------------------------------------------------------------------------!
! compute eddy viscosity
!-------------------------------------------------------------------------!
subroutine leith(nx,ny,dx,dy,w,vt)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy,sma,del,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: w
double precision, dimension (-2:nx+2,-2:ny+2)  :: vt
double precision, dimension (:,:), allocatable :: st

common /smaconstant/ sma,del
common /smag/ csd

	allocate(st(-2:nx+2,-2:ny+2))
	call vortgrad(nx,ny,dx,dy,w,st)

	csd = sma*sma*sma*del*del*del

	do j=0,ny
	do i=0,nx
	vt(i,j)=csd*st(i,j)
	end do
	end do
	
	deallocate(st)

return
end


!-------------------------------------------------------------------------!
! compute eddy viscosity
!-------------------------------------------------------------------------!
subroutine dyneddy(nx,ny,dx,dy,w,s,lp,jc,vt)
implicit none
integer :: nx,ny,idyn
double precision :: dx,dy
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: lp,jc,vt

common /dynmodel/ idyn

if(idyn.eq.1) then
	call dyn_classic(nx,ny,dx,dy,w,s,lp,jc,vt)
else if(idyn.eq.2) then
	call dyn_ba(nx,ny,dx,dy,w,s,lp,jc,vt)
else if(idyn.eq.3) then
	call dyn_ly(nx,ny,dx,dy,w,s,lp,jc,vt)
else if(idyn.eq.4) then
	call dyn_ad(nx,ny,dx,dy,w,s,lp,jc,vt)
else if(idyn.eq.5) then
	call dyn_gr(nx,ny,dx,dy,w,s,lp,vt)
end if

  
return
end

!-------------------------------------------------------------------------!
! compute eddy viscosity (dynamic gradient)
!-------------------------------------------------------------------------!
subroutine dyn_gr(nx,ny,dx,dy,w,s,lp,vt)
implicit none
integer :: i,j,nx,ny,ivis
double precision :: dx,dy,dd,nn,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: lp,vt
double precision, dimension (:,:), allocatable :: l,m,st

common /viskernel/ ivis
common /dyncoeff/ csd

allocate(l(-2:nx+2,-2:ny+2))
allocate(m(-2:nx+2,-2:ny+2))

!compute L
call grad(nx,ny,dx,dy,s,l)

!compute M
	!compute strain
	allocate(st(-2:nx+2,-2:ny+2))
    if (ivis.eq.1) then
	call strain(nx,ny,dx,dy,s,st)
    else 
    call vortgrad(nx,ny,dx,dy,w,st)
    end if

 	do j=0,ny
	do i=0,nx
	m(i,j)=st(i,j)*lp(i,j)
	end do
	end do   

	nn = 0.0d0
	dd = 0.0d0
	!compute (cs*delta)^2 =csd
	do j=0,ny
	do i=0,nx 
	nn = nn + l(i,j)*m(i,j)
	dd = dd + m(i,j)*m(i,j)
	end do
	end do
	
	!compute csd
	csd = dabs(nn/dd)

	!eddy vicosity model
	do j=0,ny
	do i=0,nx 
		vt(i,j) = csd*st(i,j)
	end do
	end do

deallocate(l,m,st)

return
end


!-------------------------------------------------------------------------!
! compute eddy viscosity (dynamic layton)
!-------------------------------------------------------------------------!
subroutine dyn_ad(nx,ny,dx,dy,w,s,lp,jc,vt)
implicit none
integer :: i,j,nx,ny,ivis
double precision :: dx,dy,dd,nn,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: lp,jc,vt
double precision, dimension (:,:), allocatable :: l,m,st

common /viskernel/ ivis
common /dyncoeff/ csd

allocate(l(-2:nx+2,-2:ny+2))
allocate(m(-2:nx+2,-2:ny+2))

!compute L
call adles(nx,ny,dx,dy,w,s,jc,l)

!compute M
	!compute strain
	allocate(st(-2:nx+2,-2:ny+2))
    if (ivis.eq.1) then
	call strain(nx,ny,dx,dy,s,st)
    else 
    call vortgrad(nx,ny,dx,dy,w,st)
    end if

 	do j=0,ny
	do i=0,nx
	m(i,j)=st(i,j)*lp(i,j)
	end do
	end do   

	nn = 0.0d0
	dd = 0.0d0
	!compute (cs*delta)^2 =csd
	do j=0,ny
	do i=0,nx 
	nn = nn + l(i,j)*m(i,j)
	dd = dd + m(i,j)*m(i,j)
	end do
	end do
	
	!compute csd
	csd = dabs(nn/dd)

	!eddy vicosity model
	do j=0,ny
	do i=0,nx 
		vt(i,j) = csd*st(i,j)
	end do
	end do

deallocate(l,m,st)

return
end


!-------------------------------------------------------------------------!
! compute eddy viscosity (dynamic layton)
!-------------------------------------------------------------------------!
subroutine dyn_ly(nx,ny,dx,dy,w,s,lp,jc,vt)
implicit none
integer :: i,j,nx,ny,ivis
double precision :: dx,dy,dd,nn,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: lp,jc,vt
double precision, dimension (:,:), allocatable :: l,m,st

common /viskernel/ ivis
common /dyncoeff/ csd

allocate(l(-2:nx+2,-2:ny+2))
allocate(m(-2:nx+2,-2:ny+2))

!compute L
call layton(nx,ny,jc,l)

!compute M
	!compute strain
	allocate(st(-2:nx+2,-2:ny+2))
    if (ivis.eq.1) then
	call strain(nx,ny,dx,dy,s,st)
    else 
    call vortgrad(nx,ny,dx,dy,w,st)
    end if

 	do j=0,ny
	do i=0,nx
	m(i,j)=st(i,j)*lp(i,j)
	end do
	end do   

	nn = 0.0d0
	dd = 0.0d0
	!compute (cs*delta)^2 =csd
	do j=0,ny
	do i=0,nx 
	nn = nn + l(i,j)*m(i,j)
	dd = dd + m(i,j)*m(i,j)
	end do
	end do
	
	!compute csd
	csd = dabs(nn/dd)

	!eddy vicosity model
	do j=0,ny
	do i=0,nx 
		vt(i,j) = csd*st(i,j)
	end do
	end do

deallocate(l,m,st)

return
end

!-------------------------------------------------------------------------!
! compute eddy viscosity (dynamic bardina)
!-------------------------------------------------------------------------!
subroutine dyn_ba(nx,ny,dx,dy,w,s,lp,jc,vt)
implicit none
integer :: i,j,nx,ny,ivis
double precision :: dx,dy,dd,nn,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: lp,jc,vt
double precision, dimension (:,:), allocatable :: l,m,st

common /viskernel/ ivis
common /dyncoeff/ csd

allocate(l(-2:nx+2,-2:ny+2))
allocate(m(-2:nx+2,-2:ny+2))

!compute L
call ss(nx,ny,dx,dy,w,s,jc,l)

!compute M
	!compute strain
	allocate(st(-2:nx+2,-2:ny+2))
    if (ivis.eq.1) then
	call strain(nx,ny,dx,dy,s,st)
    else 
    call vortgrad(nx,ny,dx,dy,w,st)
    end if

 	do j=0,ny
	do i=0,nx
	m(i,j)=st(i,j)*lp(i,j)
	end do
	end do   

	nn = 0.0d0
	dd = 0.0d0
	!compute (cs*delta)^2 =csd
	do j=0,ny
	do i=0,nx 
	nn = nn + l(i,j)*m(i,j)
	dd = dd + m(i,j)*m(i,j)
	end do
	end do
	
	!compute csd
	csd = dabs(nn/dd)

	!eddy vicosity model
	do j=0,ny
	do i=0,nx 
		vt(i,j) = csd*st(i,j)
	end do
	end do

deallocate(l,m,st)

return
end


!-------------------------------------------------------------------------!
! compute eddy viscosity (dynamic classic)
!-------------------------------------------------------------------------!
subroutine dyn_classic(nx,ny,dx,dy,w,s,lp,jc,vt)
implicit none
integer :: i,j,nx,ny,ivis
double precision :: dx,dy,kappa2,dd,nn,csd
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: lp,jc,vt
double precision, dimension (:,:), allocatable :: wf,sf,fjc,jcf,st,lwf

common /viskernel/ ivis
common /dynconstant/ kappa2
common /dyncoeff/ csd


	!compute jacobian of filtered variables
	allocate(wf(-2:nx+2,-2:ny+2))
	allocate(sf(-2:nx+2,-2:ny+2))
	allocate(fjc(-2:nx+2,-2:ny+2))
	call filter(nx,ny,w,wf)
	call filter(nx,ny,s,sf)

	call jacobian(nx,ny,dx,dy,wf,sf,fjc)

	!compute filtered jacobian
	allocate(jcf(-2:nx+2,-2:ny+2))
	call filter(nx,ny,jc,jcf)

	!compute laplacian of wf 
	allocate(lwf(-2:nx+2,-2:ny+2))
	call laplacian(nx,ny,dx,dy,wf,lwf)

	!compute strain
	allocate(st(-2:nx+2,-2:ny+2))
    if (ivis.eq.1) then
	call strain(nx,ny,dx,dy,s,st)
    else 
    call vortgrad(nx,ny,dx,dy,w,st)
    end if
	
	!get filtered st ==> sf
    call filter(nx,ny,st,sf)


	!compute |S|L on test filter ==> lwf
	do j=0,ny
	do i=0,nx
	lwf(i,j)=sf(i,j)*lwf(i,j)
	end do
	end do


	!compute |S|L ==> wf on grid filter
	do j=0,ny
	do i=0,nx
	wf(i,j)=st(i,j)*lp(i,j)
	end do
	end do

	!compute  filtered |S|L on grid filter
	call filter(nx,ny,wf,sf)

	nn = 0.0d0
	dd = 0.0d0
	!compute (cs*delta)^2 =csd
	do j=0,ny
	do i=0,nx 
	nn = nn + (fjc(i,j) - jcf(i,j))*(kappa2*lwf(i,j) - sf(i,j))
	dd = dd + (kappa2*lwf(i,j) - sf(i,j))*(kappa2*lwf(i,j) - sf(i,j))
	end do
	end do
	
	!compute csd
	csd = dabs(nn/dd)


	!limiters
	!remove negative ones, backscatter
	!if(csd.le.0.0d0) csd=0.0d0
	!limit by smagorisnky constant of 1
	!if(csd.ge.(dx*dy)) csd = dx*dy


	!eddy vicosity model
	do j=0,ny
	do i=0,nx 
		vt(i,j) = csd*st(i,j)
	end do
	end do

deallocate(wf,sf,fjc,jcf,st,lwf)



return
end

!-----------------------------------------------------------------!
!Compute vorticity gradient from stream function
!-----------------------------------------------------------------!
subroutine vortgrad(nx,ny,dx,dy,w,st)
implicit none
integer ::nx,ny,i,j
double precision::w(-2:nx+2,-2:ny+2),st(-2:nx+2,-2:ny+2),dx,dy
double precision, dimension (:), allocatable   :: a,b 
double precision, dimension (:,:), allocatable :: e,f

allocate(e(0:nx,0:ny))
allocate(f(0:nx,0:ny))


! compute strain

! wx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = w(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	e(i,j) = b(i)
	end do
end do
deallocate(a,b)

! wy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = w(i,j)
	end do
		call c4dp(a,b,dy,ny)
	do j=0,ny
	f(i,j) = b(j)
	end do
end do
deallocate(a,b)


!strain
do j=0,ny
do i=0,nx
st(i,j) = dsqrt(f(i,j)*f(i,j) + e(i,j)*e(i,j))
end do
end do


deallocate(e,f)


return
end 

!-----------------------------------------------------------------!
!Compute strain from stream function
!-----------------------------------------------------------------!
subroutine strain(nx,ny,dx,dy,s,st)
implicit none
integer ::nx,ny,i,j
double precision::s(-2:nx+2,-2:ny+2),st(-2:nx+2,-2:ny+2),dx,dy
double precision, dimension (:), allocatable   :: a,b 
double precision, dimension (:,:), allocatable :: e,f

allocate(e(0:nx,0:ny))
allocate(f(0:nx,0:ny))


! compute strain

! sx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = s(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	e(i,j) = b(i)
	end do
end do
deallocate(a,b)

! sx_y
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = e(i,j)
	end do
		call c4dp(a,b,dy,ny)
	do j=0,ny
	f(i,j) = b(j)
	end do
end do
deallocate(a,b)



! sxx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = s(i,j)
	end do
		call c4ddp(a,b,dx,nx)
	do i=0,nx
	e(i,j) = b(i)
	end do
end do
deallocate(a,b)

! syy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4ddp(a,b,dy,ny)
	do j=0,ny
	e(i,j) = e(i,j) - b(j)
	end do
end do
deallocate(a,b)


!strain
do j=0,ny
do i=0,nx
st(i,j) = dsqrt(4.0d0*f(i,j)*f(i,j) + e(i,j)*e(i,j))
end do
end do


deallocate(e,f)


return
end 

!-------------------------------------------------------------------------!
! compute laplacian
! Modified second order - Romit 
!-------------------------------------------------------------------------!
subroutine laplacian(nx,ny,dx,dy,u,lp)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy, d2wdy2, d2wdx2
double precision, dimension (-2:nx+2,-2:ny+2)  :: u,lp
double precision, dimension (:), allocatable   :: a,b 



do j = 0,ny
	do i = 0,nx
		d2wdy2 = (u(i, j+1) + u(i, j-1) - 2.0 * u(i, j)) / (dy * dy)
		d2wdx2 = (u(i+1, j) + u(i-1, j) - 2.0 * u(i, j)) / (dx * dx)

		lp(i,j) = d2wdx2 + d2wdy2

	end do
end do



! uxx
! allocate(a(0:nx),b(0:nx))
! do j=0,ny
! 	do i=0,nx
! 	a(i) = u(i,j)
! 	end do
! 		call c4ddp(a,b,dx,nx)
! 	do i=0,nx
! 	lp(i,j) = b(i)
! 	end do
! end do
! deallocate(a,b)

! ! uyy
! allocate(a(0:ny),b(0:ny))
! do i=0,nx
! 	do j=0,ny
! 	a(j) = u(i,j)
! 	end do
! 		call c4ddp(a,b,dy,ny)
! 	do j=0,ny
! 	lp(i,j) = lp(i,j) + b(j)
! 	end do
! end do
! deallocate(a,b)


return
end

!-------------------------------------------------------------------------!
! compute jacobian 
!-------------------------------------------------------------------------!
subroutine jacobian(nx,ny,dx,dy,w,s,jc)
implicit none
integer :: nx,ny,isolver
double precision :: dx,dy
double precision, dimension (0:nx,0:ny)  :: w,s,jc

common /solver/ isolver

if (isolver.eq.1) then
call jacobian_compact(nx,ny,dx,dy,w,s,jc)
else
call jacobian_arakawa(nx,ny,dx,dy,w,s,jc)
end if


return
end

!-------------------------------------------------------------------------!
! compute jacobian (compact)
!-------------------------------------------------------------------------!
subroutine jacobian_compact(nx,ny,dx,dy,w,s,jc)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s,jc
double precision, dimension (:), allocatable   :: a,b 
double precision, dimension (:,:), allocatable :: e

! jacobian (convective term):
allocate(e(0:nx,0:ny))

! sy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4dp(a,b,dy,ny)       
	do j=0,ny
	e(i,j) = b(j)
	end do
end do
deallocate(a,b)

! wx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = w(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	jc(i,j) = e(i,j)*b(i)
	end do
end do
deallocate(a,b)



! sx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = s(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	e(i,j) = b(i)
	end do
end do
deallocate(a,b)

! wy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = w(i,j)
	end do
		call c4dp(a,b,dy,ny)
	do j=0,ny
	jc(i,j) = jc(i,j) - e(i,j)*b(j)
	end do
end do
deallocate(a,b)
deallocate(e)

call bc(nx,ny,jc)

return
end


!-------------------------------------------------------------------------!
! compute jacobian by second order Arakawa scheme (conservative)
! Modified Romit Maulik
!-------------------------------------------------------------------------!
subroutine jacobian_arakawa(nx,ny,dx,dy,w,s,jc)
implicit none
integer :: i,j,nx,ny
double precision :: dx,dy,j1,j2,j3,j11,j22,j33,g,e,z,h
double precision, dimension (-2:nx+2,-2:ny+2)  :: w,s
double precision, dimension (-2:nx+2,-2:ny+2)  :: jc

g = 1.0d0/(4.0d0*dx*dy)
e = 1.0d0/(8.0d0*dx*dy)
z = 2.0d0/3.0d0
h = 1.0d0/3.0d0

call bc(nx,ny,w)
call bc(nx,ny,s)

! do j=0,ny
! do i=0,nx

! j1 = g*((w(i+1,j)-w(i-1,j))*(s(i,j+1)-s(i,j-1)) &
!        -(w(i,j+1)-w(i,j-1))*(s(i+1,j)-s(i-1,j)) )

! j2 = g*(w(i+1,j)*(s(i+1,j+1)-s(i+1,j-1)) &
!        -w(i-1,j)*(s(i-1,j+1)-s(i-1,j-1)) &
! 	   -w(i,j+1)*(s(i+1,j+1)-s(i-1,j+1)) &
! 	   +w(i,j-1)*(s(i+1,j-1)-s(i-1,j-1)) )

! j3 = g*(w(i+1,j+1)*(s(i,j+1)-s(i+1,j)) &
!        -w(i-1,j-1)*(s(i-1,j)-s(i,j-1)) &
! 	   -w(i-1,j+1)*(s(i,j+1)-s(i-1,j)) &
! 	   +w(i+1,j-1)*(s(i+1,j)-s(i,j-1)) )


! j11= e*((w(i+1,j+1)-w(i-1,j-1))*(s(i-1,j+1)-s(i+1,j-1)) &
!        -(w(i-1,j+1)-w(i+1,j-1))*(s(i+1,j+1)-s(i-1,j-1)) )

! j22= e*(w(i+1,j+1)*(s(i,j+2)-s(i+2,j)) &
!        -w(i-1,j-1)*(s(i-2,j)-s(i,j-2)) &
! 	   -w(i-1,j+1)*(s(i,j+2)-s(i-2,j)) &
! 	   +w(i+1,j-1)*(s(i+2,j)-s(i,j-2)) )


! j33= e*(w(i+2,j)*(s(i+1,j+1)-s(i+1,j-1)) &
!        -w(i-2,j)*(s(i-1,j+1)-s(i-1,j-1)) &
! 	   -w(i,j+2)*(s(i+1,j+1)-s(i-1,j+1)) &
! 	   +w(i,j-2)*(s(i+1,j-1)-s(i-1,j-1)) )


! jc(i,j) = (j1+j2+j3)*z - (j11+j22+j33)*h

! end do
! end do

do j = 0,ny
do i = 0,nx


j1 = 1.0/(4.0*dx*dy) * ((w(i+1,j)-w(i-1,j)) * (s(i,j+1) - s(i,j-1)) &
			- (w(i,j+1)-w(i,j-1)) * (s(i+1,j) - s(i-1,j)))

j2 = 1.0 / (4.0 * dx * dy) * (w(i+1, j) * (s(i+1, j+1) - s(i+1, j-1)) &
                                         - w(i-1, j) * (s(i-1, j+1) - s(i-1, j-1)) &
                                         - w(i, j+1) * (s(i+1, j+1) - s(i-1, j+1)) &
                                         + w(i, j-1) * (s(i+1, j-1) - s(i-1, j-1)) &
                                          )

j3 = 1.0 / (4.0 * dx * dy) * (w(i+1, j+1) * (s(i, j+1) - s(i+1, j)) &
                                        -  w(i-1, j-1) * (s(i-1, j) - s(i, j-1)) &
                                        -  w(i-1, j+1) * (s(i, j+1) - s(i-1, j)) &
                                        +  w(i+1, j-1) * (s(i+1, j) - s(i, j-1)) &
                                          )

jc(i, j) = (j1 + j2 + j3)/3.0

end do
end do

call bc(nx,ny,jc)

return
end


!-----------------------------------------------------------------!
! computes integral values
!-----------------------------------------------------------------!
subroutine history(nx,ny,dx,dy,dt,w,cfl,en,ew)
implicit none
integer ::nx,ny,i,j
double precision::dx,dy,dt,cfl,en,ew,w(-2:nx+2,-2:ny+2)
double precision,dimension(:),allocatable  :: a,b 
double precision,dimension(:,:),allocatable:: s,u,v,g 
double precision::umax,umin,vmax,vmin,area,pi,cflc,dta,dtv

common /adaptive/ cflc,dta,dtv

pi = 4.0d0*datan(1.0d0)
area = (2.0d0*pi)**2


!compute stream function
allocate(s(-2:nx+2,-2:ny+2))
call fps4(nx,ny,dx,dy,-w,s)

allocate(u(0:nx,0:ny),v(0:nx,0:ny))

! u = sy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4dp(a,b,dy,ny)
	do j=0,ny
	u(i,j) = b(j)
	end do
end do
deallocate(a,b)

! v=-sx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = s(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	v(i,j) =-b(i)
	end do
end do
deallocate(a,b)

umax = maxval(u)
umin = minval(u)
vmax = maxval(v)
vmin = minval(v)

umax = max(dabs(umax),dabs(umin))
vmax = max(dabs(vmax),dabs(vmin))

cfl = max(umax*dt/dx, vmax*dt/dy)

dta = min(cflc*dx/umax,cflc*dy/vmax,dtv)


!compute total energy
allocate(g(0:nx,0:ny))
do i=0,nx
do j=0,ny
g(i,j)=0.5d0*(u(i,j)**2 + v(i,j)**2)
end do
end do
call simp2D(nx,ny,dx,dy,g,en)
deallocate(g)
en = en/area


!compute total enstrophy
allocate(g(0:nx,0:ny))
do i=0,nx
do j=0,ny
g(i,j)=0.5d0*(w(i,j)**2)
end do
end do
call simp2D(nx,ny,dx,dy,g,ew)
deallocate(g)
ew = ew/area

deallocate(u,v,s)

return
end 


!-----------------------------------------------------------------!
subroutine saved(nx,ny,time,w)
implicit none
integer ::nx,ny,i,j
double precision::time,w(-2:nx+2,-2:ny+2)

	open(3,file='final.dat')
	write(3,*)time
	write(3,*)nx,ny
	write(3,*)((w(i,j), i=0,nx), j=0,ny)
	close(3)

return
end 

!-----------------------------------------------------------------!
subroutine outw(i1,nx,ny,dx,dy,time,w)
implicit none
integer ::nx,ny,i1,i,j
double precision::dx,dy,time,w(-2:nx+2,-2:ny+2),wmax

open(50000+i1)
write(50000+i1,*) 'variables ="x","y","w"'
write(50000+i1,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',time,'"'
do j=0,ny
do i=0,nx
write(50000+i1,52) dfloat(i)*dx,dfloat(j)*dy,w(i,j)
end do
end do
close(50000+i1)

! wmax = maxval(w)
! open(20000+i1)
! write(20000+i1,*) 'variables ="x","y","w"'
! write(20000+i1,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',time,'"'
! do j=0,ny
! do i=0,nx
! write(20000+i1,52) dfloat(i)*dx,dfloat(j)*dy,w(i,j)/wmax
! end do
! end do
! close(20000+i1)


51 format(a16,i8,a4,i8,a10,f10.4,a3)
52 format(3e16.6)
return
end 


!-----------------------------------------------------------------!
subroutine spec(i2,nx,ny,time,w)
implicit none
integer ::nx,ny,i2
double precision::w(-2:nx+2,-2:ny+2)
double precision::time,pi
integer::i,j,k,n,ic
double precision::kx(0:nx-1),ky(0:ny-1),kk
double precision,parameter:: tiny=1.0d-10
double precision,dimension(:),allocatable:: data1d,en
double precision,dimension(:,:),allocatable::es
integer,parameter::ndim=2
integer::nn(ndim),isign

allocate(data1d(2*nx*ny))

pi = 4.0d0*datan(1.0d0)

nn(1)= nx
nn(2)= ny

!finding fourier coefficients of w 
!invese fourier transform
!find the vorticity in Fourier space
k=1
do j=0,ny-1  
do i=0,nx-1   
	data1d(k)   =  w(i,j)
	data1d(k+1) =  0.0d0    
k = k + 2
end do
end do
!normalize
do k=1,2*nx*ny
data1d(k)=data1d(k)/dfloat(nx*ny)
end do
!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)


!Wave numbers 
do i=0,nx/2-1
kx(i)      = dfloat(i)
kx(i+nx/2) = dfloat(i-nx/2)
end do
kx(0) = tiny

do j=0,ny/2-1
ky(j)      = dfloat(j)
ky(j+ny/2) = dfloat(j-ny/2)
end do
ky(0) = tiny

!Energy spectrum (for all wavenumbers)
allocate(es(0:nx-1,0:ny-1))
k=1
do j=0,ny-1
do i=0,nx-1 
kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
es(i,j) = pi*(data1d(k)*data1d(k) + data1d(k+1)*data1d(k+1))/kk
k = k + 2
end do
end do

open(600+i2)
write(600+i2,*)'variables ="k","E(k)"'
write(600+i2,54)'zone f=point i=',(nx-1)*(ny-1),',t="time',time,'"'
do j=1,ny-1
do i=1,nx-1
kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
write(600+i2,103) kk,es(i,j)
end do
end do
close(600+i2)


!Plot angle avaraged energy spectrum
n = nint(0.5d0*dsqrt(dfloat(nx*nx + ny*ny)))-1
allocate(en(n))

do k=1,n
en(k) = 0.0d0
ic = 0
do j=1,ny-1
do i=1,nx-1
kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
    if(kk.ge.(dfloat(k)-0.5d0).and.kk.le.(dfloat(k)+0.5d0)) then
    ic = ic + 1
    en(k) = en(k) + es(i,j)
    end if
end do
end do
en(k) = en(k) / dfloat(ic)
end do

open(700+i2)
write(700+i2,*)'variables ="k","E(k)"'
write(700+i2,54)'zone f=point i=',n,',t="time',time,'"'
!write(77,54)'zone f=point i=',n,',t="time',time,'"'
do k=1,n
write(700+i2,103)dfloat(k),en(k)
!write(77,103)dfloat(k),en(k)
end do
close(700+i2)


deallocate(data1d,es,en)

103 format(2e16.6)
54 format(a16,i8,a10,f10.4,a3)
return
end 



!-----------------------------------------------------------------!
subroutine line(i3,nx,ny,dx,dy,time,w)
implicit none
integer ::nx,ny,i3,i,j
double precision::dx,dy,time,w(-2:nx+2,-2:ny+2)

open(300+i3)
write(300+i3,*) 'variables ="y","w"'
write(300+i3,54)'zone f=point i=',ny+1,',t="time',time,'"'
do j=0,ny
write(300+i3,55) dfloat(j)*dy,w(nx/2,j)
end do
close(300+i3)

open(400+i3)
write(400+i3,*) 'variables ="x","w"'
write(400+i3,54)'zone f=point i=',nx+1,',t="time',time,'"'
do i=0,nx
write(400+i3,55) dfloat(i)*dx,w(i,ny/2)
end do
close(400+i3)


54 format(a16,i8,a10,f10.4,a3)
55 format(2e16.6)
return
end 

!-----------------------------------------------------------------!
!Compute structure functions
!-----------------------------------------------------------------!
subroutine strfun(i4,nx,ny,time,w)
implicit none
integer::nx,ny,ic,p,i4,i,j
double precision::time
double precision::w(-2:nx+2,-2:ny+2),dw2(nx/2)

do p=1,nx/2

	ic=0
	dw2(p)=0.0d0		
	!computing x directional vorticity difference
	do j=0,ny
	do i=0,nx-p	
		ic=ic+1		
		dw2(p)=dw2(p) + (w(i+p,j)-w(i,j))**2
	end do
	end do

	!computing y directional vorticity difference
	do i=0,nx
	do j=0,ny-p	
		ic = ic+1		
		dw2(p)=dw2(p) + (w(i,j+p)-w(i,j))**2
	end do
	end do 

	!normalizing (ensamble averaging)
	dw2(p)=dw2(p)/dfloat(ic)

end do


!writing
open(800+i4)
write(800+i4,*) 'variables ="r","SF2"'
write(800+i4,54)'zone f=point i=',nx/2,',t="time',time,'"'
do i=1,nx/2
write(800+i4,*)i,dw2(i)
end do
close(800+i4)

54 format(a16,i8,a10,f10.4,a3)
return
end 



!-----------------------------------------------------------------!
! fft routine from numerical recipes
! ndim: dimension of the transform (i.e.; 2 for 2d problems)
! nn  : number of points in each direction
! data: one-dimensional array including real and imaginary part 
!-----------------------------------------------------------------!
subroutine fourn(data,nn,ndim,isign)
implicit none
integer:: ndim,isign
integer:: nn(ndim)
real*8:: data(*)
real*8:: wr,wi,wpr,wpi,wtemp,theta,tempr,tempi
integer::ntot,n,nrem,nprev,idim,ip1,ip2,ip3,i1,i2,i3
integer::i2rev,i3rev,ibit,ifp1,ifp2,k1,k2

      ntot=1
      do 11 idim=1,ndim
        ntot=ntot*nn(idim)
11    continue
      nprev=1
      do 18 idim=1,ndim
        n=nn(idim)
        nrem=ntot/(n*nprev)
        ip1=2*nprev
        ip2=ip1*n
        ip3=ip2*nrem
        i2rev=1
        do 14 i2=1,ip2,ip1
          if(i2.lt.i2rev)then
            do 13 i1=i2,i2+ip1-2,2
              do 12 i3=i1,ip3,ip2
                i3rev=i2rev+i3-i2
                tempr=data(i3)
                tempi=data(i3+1)
                data(i3)=data(i3rev)
                data(i3+1)=data(i3rev+1)
                data(i3rev)=tempr
                data(i3rev+1)=tempi
12            continue
13          continue
          endif
          ibit=ip2/2
1         if ((ibit.ge.ip1).and.(i2rev.gt.ibit)) then
            i2rev=i2rev-ibit
            ibit=ibit/2
          go to 1
          endif
          i2rev=i2rev+ibit
14      continue
        ifp1=ip1
2       if(ifp1.lt.ip2)then
          ifp2=2*ifp1
          theta=isign*6.28318530717959d0/(ifp2/ip1)
          wpr=-2.d0*dsin(0.5d0*theta)**2
          wpi=dsin(theta)
          wr=1.d0
          wi=0.d0
          do 17 i3=1,ifp1,ip1
            do 16 i1=i3,i3+ip1-2,2
              do 15 i2=i1,ip3,ifp2
                k1=i2
                k2=k1+ifp1
                tempr=sngl(wr)*data(k2)-sngl(wi)*data(k2+1)
                tempi=sngl(wr)*data(k2+1)+sngl(wi)*data(k2)
                data(k2)=data(k1)-tempr
                data(k2+1)=data(k1+1)-tempi
                data(k1)=data(k1)+tempr
                data(k1+1)=data(k1+1)+tempi
15            continue
16          continue
            wtemp=wr
            wr=wr*wpr-wi*wpi+wr
            wi=wi*wpr+wtemp*wpi+wi
17        continue
          ifp1=ifp2
        go to 2
        endif
        nprev=n*nprev
18    continue

return
end


!---------------------------------------------------------------------------!
!Spectral accurate Poisson solver
!Periodic, equidistant grid
!Taken from MAE5093 - Github
!Matches with GS
!---------------------------------------------------------------------------!
subroutine fps4(nx,ny,dx,dy,f_org,u_org)
implicit none
integer,intent(in)::nx,ny
double precision,intent(in) ::dx,dy
double precision,intent(in)::f_org(-2:nx+2,-2:ny+2)
double precision,intent(inout):: u_org(-2:nx+2,-2:ny+2)
double precision ::pi,Lx,Ly,den
double precision ::kx(0:nx-1),ky(0:ny-1) 
double precision ::data1d(2*nx*ny) 
integer::i,j,k,isign,ndim,nn(2)

!2d data
ndim =2
nn(1)=nx
nn(2)=ny

!1.Find the f coefficient in Fourier space
!assign 1d data array
k=1
do j=0,ny-1  
do i=0,nx-1   
  data1d(k)   =  f_org(i,j)
  data1d(k+1) =  0.0d0    
k = k + 2
end do
end do

!normalize
do k=1,2*nx*ny
data1d(k)=data1d(k)/dfloat(nx*ny)
end do
!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)

!2.Solve for u coeeficient in Fourier space
!coefficients
Lx = dfloat(nx)*dx
Ly = dfloat(ny)*dy

!wave numbers (scaled)
pi = 4.0d0*datan(1.0d0)
do i=0,nx/2-1
kx(i)      = (2.0d0*pi/Lx)*dfloat(i)
kx(i+nx/2) = (2.0d0*pi/Lx)*dfloat(i-nx/2)
end do
do j=0,ny/2-1
ky(j)      = (2.0d0*pi/Ly)*dfloat(j)
ky(j+ny/2) = (2.0d0*pi/Ly)*dfloat(j-ny/2)
end do
kx(0) = 1.0d-6 !to eleminate zero division
ky(0) = 1.0d-6 !to eleminate zero division
data1d(1) = 0.0d0
data1d(2) = 0.0d0

!Fourier coefficients for u
k=1
do j=0,ny-1
do i=0,nx-1   
    den = -(kx(i)*kx(i))-(ky(j)*ky(j))
  data1d(k)   =  data1d(k)/den
  data1d(k+1) =  data1d(k+1)/den
k = k + 2
end do
end do

!3. Find u values on physical space
!forward fourier transform
isign= 1
call fourn(data1d,nn,ndim,isign)

!assign 2d array
k=1
do j=0,ny-1
do i=0,nx-1
u_org(i,j)=data1d(k)
k=k+2
end do
end do


! periodicity
do i=0,nx-1
u_org(i,ny)=u_org(i,0)
end do
do j=0,ny-1
u_org(nx,j)=u_org(0,j)
end do
u_org(nx,ny)=u_org(0,0)

!extend for periodic b.c.
call bc(nx,ny,u_org)

return
end

! !-----------------------------------------------------------------------------------------!
! !fast Poisson solver in 2D periodic domain (forth-order)
! !-----------------------------------------------------------------------------------------!
! subroutine fps4(nx,ny,dx,dy,f,u)
! implicit none
! integer::nx,ny
! real*8 ::f(-2:nx+2,-2:ny+2),u(-2:nx+2,-2:ny+2)
! real*8 ::dx,dy
! real*8 ::data1d(2*nx*ny)
! integer,parameter::ndim=2
! integer::nn(ndim),isign
! real*8 ::pi,hx,hy,ra,aa,bb,cc,dd,ee,nom,den
! integer::i,j,k
! real*8,parameter:: eps=1.0d-10

! nn(1)= nx
! nn(2)= ny

! pi = 4.0d0*datan(1.0d0)
! ra = dx/dy
! hx = 2.0d0*pi/dfloat(nx)
! hy = 2.0d0*pi/dfloat(ny)
! aa =-10.0d0*(1.0d0 + ra*ra )
! bb = 5.0d0-ra*ra
! cc = 5.0d0*ra*ra -1.0d0
! dd = 0.5d0*(1.0d0 +ra*ra)
! ee = (dx*dx)/2.0d0

! !step-1
! !finding fourier coefficients of f 
! !invese fourier transform
! k=1
! do j=0,ny-1
! do i=0,nx-1
! data1d(k)  =f(i,j)
! data1d(k+1)=0.0d0
! k=k+2
! end do
! end do

! isign=-1
! call fourn(data1d,nn,ndim,isign)

! do j=1,2*nx*ny
! data1d(j)=data1d(j)/dfloat(nx*ny)
! end do

! !step-2
! !find fourier coefficients of u
! !algebraic equation
! k=1
! do j=0,ny-1
! do i=0,nx-1
! nom = data1d(k)*ee*(8.0d0 + 2.0d0*dcos(hx*dfloat(i)) + 2.0d0*dcos(hy*dfloat(j)))
! den = aa + 2.0d0*bb*dcos(hx*dfloat(i)) + 2.0d0*cc*dcos(hy*dfloat(j)) + 4.0d0*dd*dcos(hx*dfloat(i))*dcos(hy*dfloat(j)) + eps
! data1d(k)=nom/den
! data1d(k+1)=nom/den
! k=k+2
! end do 
! end do


! !step-3
! !find the u
! !forward fourier transform
! isign= 1
! call fourn(data1d,nn,ndim,isign)

! k=1
! do j=0,ny-1
! do i=0,nx-1
! u(i,j)=data1d(k)
! k=k+2
! end do
! end do

! ! periodicity
! do i=0,nx-1
! u(i,ny)=u(i,0)
! end do
! do j=0,ny-1
! u(nx,j)=u(0,j)
! end do
! u(nx,ny)=u(0,0)

! !extend for periodic b.c.
! call bc(nx,ny,u)
    
! return
! end



!------------------------------------------------------------------!
! compact interpolations for derivarives
!------------------------------------------------------------------!

!------------------------------------------------------------------!
! c4dp:  4th-order compact scheme for first-degree derivative(up)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c4dp(u,up,h,n)
implicit none
integer :: n,i
double precision   :: h,alpha,beta
double precision , dimension (0:n)  :: u,up
double precision , dimension (0:n-1):: a,b,c,r,x 

do i=0,n-1
a(i) = 1.0d0/4.0d0
b(i) = 1.0d0
c(i) = 1.0d0/4.0d0
end do

do i=1,n-1
r(i) = 3.0d0/2.0d0*(u(i+1)-u(i-1))/(2.0d0*h) 
end do
r(0) = 3.0d0/2.0d0*(u(1)-u(n-1))/(2.0d0*h) 
 
alpha = 1.0d0/4.0d0
beta  = 1.0d0/4.0d0

call ctdms(a,b,c,alpha,beta,r,x,0,n-1) 

do i=0,n-1
up(i)=x(i)
end do
up(n)=up(0)

return
end

!------------------------------------------------------------------!
! c6dp:  6th-order compact scheme for first-degree derivative(up)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c6dp(u,up,h,n)
implicit none
integer :: n,i
double precision   :: h,alpha,beta
double precision , dimension (0:n)  :: u,up
double precision , dimension (0:n-1):: a,b,c,r,x 

do i=0,n-1
a(i) = 1.0d0/3.0d0
b(i) = 1.0d0
c(i) = 1.0d0/3.0d0
end do

do i=2,n-2
r(i) = 14.0d0/9.0d0*(u(i+1)-u(i-1))/(2.0d0*h) &
     + 1.0d0/9.0d0*(u(i+2)-u(i-2))/(4.0d0*h)
end do
r(1) = 14.0d0/9.0d0*(u(2)-u(0))/(2.0d0*h) &
     + 1.0d0/9.0d0*(u(3)-u(n-1))/(4.0d0*h) 

r(n-1) = 14.0d0/9.0d0*(u(n)-u(n-2))/(2.0d0*h) &
     + 1.0d0/9.0d0*(u(1)-u(n-3))/(4.0d0*h)

r(0) = 14.0d0/9.0d0*(u(1)-u(n-1))/(2.0d0*h) &
     + 1.0d0/9.0d0*(u(2)-u(n-2))/(4.0d0*h)
     
alpha = 1.0d0/3.0d0
beta  = 1.0d0/3.0d0

call ctdms(a,b,c,alpha,beta,r,x,0,n-1) 

do i=0,n-1
up(i)=x(i)
end do
up(n)=up(0)

return
end

!------------------------------------------------------------------!
! c4ddp: 4th-order compact scheme for second-degree derivative(upp)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c4ddp(u,upp,h,n)
implicit none
integer :: n,i
double precision   :: h,alpha,beta
double precision , dimension (0:n)  :: u,upp
double precision , dimension (0:n-1):: a,b,c,r,x 

do i=0,n-1
a(i) = 1.0d0/10.0d0
b(i) = 1.0d0
c(i) = 1.0d0/10.0d0
end do

do i=1,n-1
r(i) = 6.0d0/5.0d0*(u(i-1)-2.0d0*u(i)+u(i+1))/(h*h) 
end do
r(0) = 6.0d0/5.0d0*(u(n-1)-2.0d0*u(0)+u(1))/(h*h) 
 
alpha = 1.0d0/10.0d0
beta  = 1.0d0/10.0d0

call ctdms(a,b,c,alpha,beta,r,x,0,n-1) 

do i=0,n-1
upp(i)=x(i)
end do
upp(n)=upp(0)

return
end

!------------------------------------------------------------------!
! c6ddp: 6th-order compact scheme for second-degree derivative(upp)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c6ddp(u,upp,h,n)
implicit none
integer :: n,i
double precision   :: h,alpha,beta
double precision , dimension (0:n)  :: u,upp
double precision , dimension (0:n-1):: a,b,c,r,x 


do i=0,n-1
a(i) = 2.0d0/11.0d0
b(i) = 1.0d0
c(i) = 2.0d0/11.0d0
end do

do i=2,n-2
r(i) = 12.0d0/11.0d0*(u(i-1)-2.0d0*u(i)+u(i+1))/(h*h) &
     + 3.0d0/11.0d0*(u(i-2)-2.0d0*u(i)+u(i+2))/(4.0d0*h*h) 
end do
r(1) = 12.0d0/11.0d0*(u(0)-2.0d0*u(1)+u(2))/(h*h) &
     + 3.0d0/11.0d0*(u(n-1)-2.0d0*u(1)+u(3))/(4.0d0*h*h)  

r(n-1) = 12.0d0/11.0d0*(u(n-2)-2.0d0*u(n-1)+u(n))/(h*h) &
     + 3.0d0/11.0d0*(u(n-3)-2.0d0*u(n-1)+u(1))/(4.0d0*h*h) 

r(0) = 12.0d0/11.0d0*(u(n-1)-2.0d0*u(0)+u(1))/(h*h) &
     + 3.0d0/11.0d0*(u(n-2)-2.0d0*u(0)+u(2))/(4.0d0*h*h) 
     
alpha = 2.0d0/11.0d0
beta  = 2.0d0/11.0d0

call ctdms(a,b,c,alpha,beta,r,x,0,n-1) 

do i=0,n-1
upp(i)=x(i)
end do
upp(n)=upp(0)

return
end

!----------------------------------------------------!
! solution tridiagonal systems (regular tri-diagonal)
! a:subdiagonal, b: diagonal, c:superdiagonal
! r:rhs, u:results
! s: starting index
! e: ending index
!----------------------------------------------------!
subroutine tdms(a,b,c,r,u,s,e)
implicit none 
integer::s,e
double precision ::a(s:e),b(s:e),c(s:e),r(s:e),u(s:e) 
integer::j  
double precision ::bet,gam(s:e) 
bet=b(s)  
u(s)=r(s)/bet  
do j=s+1,e  
gam(j)=c(j-1)/bet  
bet=b(j)-a(j)*gam(j)  
u(j)=(r(j)-a(j)*u(j-1))/bet  
end do  
do j=e-1,s,-1  
u(j)=u(j)-gam(j+1)*u(j+1)  
end do  
return  
end  


!-------------------------------------------------------------------!
! solution of cyclic tridiagonal systems (periodic tridiagonal)
! n:matrix size (starting from 1)
! a:subdiagonal, b: diagonal, c:superdiagonal
! r:rhs, x:results
! alpha:sub entry (first value in e-th eq.)
! beta: super entry (last value in s-th eq.)
!-------------------------------------------------------------------!
subroutine ctdms(a,b,c,alpha,beta,r,x,s,e) 
implicit none
integer:: s,e
double precision :: alpha,beta,a(s:e),b(s:e),c(s:e),r(s:e),x(s:e)  
integer:: i  
double precision :: fact,gamma,bb(s:e),u(s:e),z(s:e)
if((e-s).le.2) then
write(*,*) ' matrix too small in cyclic' 
stop
end if 
gamma=-b(s)  
bb(s)=b(s)-gamma  
bb(e)=b(e)-alpha*beta/gamma  
do i=s+1,e-1  
bb(i)=b(i)  
end do  
call tdms(a,bb,c,r,x,s,e) 	      
u(s)=gamma  
u(e)=alpha  
do i=s+1,e-1  
u(i)=0.0d0  
end do  
call tdms(a,bb,c,u,z,s,e) 
fact=(x(s)+beta*x(e)/gamma)/(1.0d0+z(s)+beta*z(e)/gamma)  
do i=s,e  
x(i)=x(i)-fact*z(i)  
end do  
return  
end


!----------------------------------------------------------!
!Simpson's 1/3 rule for numerical integration of g(i,j)
!for equally distributed mesh with interval dx and dy
!n should be power of 2
!dual integration
!----------------------------------------------------------!
subroutine simp2D(nx,ny,dx,dy,g,s)
implicit none
integer::nx,ny,i,j,nh
real*8 ::dx,dy,g(0:nx,0:ny),s,ds,th
real*8,allocatable::sy(:)

allocate(sy(0:ny))

	nh = int(nx/2)
	th = 1.0d0/3.0d0*dx
    
do j=0,ny
	sy(j) = 0.0d0
	do i=0,nh-1
	ds = th*(g(2*i,j)+4.0d0*g(2*i+1,j)+g(2*i+2,j))
	sy(j) = sy(j) + ds
	end do
end do

	nh = int(ny/2)	
	th = 1.0d0/3.0d0*dy
    
	s = 0.0d0
	do j=0,nh-1
	ds = th*(sy(2*j)+4.0d0*sy(2*j+1)+sy(2*j+2))
	s = s + ds
	end do

deallocate(sy)
return
end
