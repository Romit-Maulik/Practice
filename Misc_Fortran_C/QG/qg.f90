!------------------------------------------------------------------------------!
!							<<< 2D QG Solver >>>
!------------------------------------------------------------------------------!
!2D Unsteady Vorticity-Stream Function Formulation + double gyre forcing
!Re,Ro are main dimensionless parameters
!Free slip boundary conditions for velocity field (zero vorticity on boundaries)
!Drichlet bc for stream function in Poisson equation
!Arakawa schemes are implemented
!------------------------------------------------------------------------------!
!Omer San, January 2013, Virginia Tech, omersan@vt.edu
!Romit Maulik, September 2018, Okstate, romit.maulik@okstate.edu
!------------------------------------------------------------------------------!
program qg
implicit none
integer::i,j,k,nx,ny,nt,isc,ich,icount,iturb
real*8 ::Lx,Ly,Tmax,Tave,dx,dy,dt,time,t1,t2,tt1,tt2,tt
real*8,allocatable::x(:),y(:),f(:,:)
real*8,allocatable::s(:,:),w(:,:)
real*8,allocatable::sa(:,:),wa(:,:)
real*8 ::Re,Ro,St,pi,tote,cfl,vm
integer::ffile,nfile,nsnap,fsnap,isnap,fhist

common /phys/ Re,Ro,St
common /turb/ iturb

!read input file
open(10,file='input.txt')
read(10,*)nx
read(10,*)ny
read(10,*)Lx
read(10,*)Ly
read(10,*)Tmax
read(10,*)Tave
read(10,*)dt
read(10,*)Re
read(10,*)Ro
read(10,*)St
read(10,*)nfile
read(10,*)nsnap
read(10,*)fhist
read(10,*)isc
read(10,*)ich
read(10,*)iturb
close(10)



!check input file
if(ich.ne.19) then
print*,"*** check input file ***"
stop
end if

!grid step size
dx=Lx/dfloat(nx)
dy=Ly/dfloat(ny)

!totat time step
nt=nint(Tmax/dt)

!coordinates
allocate(x(0:nx))
allocate(y(0:ny))
do i=0,nx
x(i)=dfloat(i)*dx
end do
do j=0,ny
y(j)=-1.0d0 + dfloat(j)*dy
end do

!allocate arrays
allocate(s(0:nx,0:ny))
allocate(w(0:nx,0:ny))
allocate(f(0:nx,0:ny))

call cpu_time(t1)
tt = 0.0d0

time = 0.0d0

!initial conditions
do j=0,ny
do i=0,nx
s(i,j) = 0.0d0
w(i,j) = 0.0d0
end do
end do

pi=4.0d0*datan(1.0d0)
!forcing function
do j=0,ny
do i=0,nx
f(i,j) = (1.0d0/Ro)*dsin(pi*y(j))
end do
end do

ffile= int(nt/nfile)
fsnap= int(nt/nsnap)

open(500,file='field.plt')
write(500,*) 'variables ="x","y","s","w"'
call outfield(nx,ny,time,x,y,s,w)

isnap= 0
!snapshot data
open(8000+isnap)
write(8000+isnap,*)((s(i,j),i=0,nx),j=0,ny)
close(8000+isnap)    
open(9000+isnap)
write(9000+isnap,*)((w(i,j),i=0,nx),j=0,ny)
close(9000+isnap)

open(600,file='history.plt')
write(600,*) 'variables ="t","tote","cfl","vm"'

!mean fields
icount=0
allocate(sa(0:nx,0:ny))
allocate(wa(0:nx,0:ny))
do j=0,ny
do i=0,nx
sa(i,j) = 0.0d0
wa(i,j) = 0.0d0
end do
end do
    
!Time integration
do k=1,nt
time = time+dt
	!solver
    call cpu_time(tt1)
	call tvdrk3(nx,ny,dx,dy,dt,s,w,f)
    call cpu_time(tt2)
    tt = tt + (tt2-tt1)
    
	
	if (mod(k,fhist).eq.0) then
    call history(nx,ny,dx,dy,dt,s,cfl,tote,vm)
    write(600,*)time,tote,cfl,vm
    end if

	!output
	if(mod(k,ffile).eq.0) then
	call outfield(nx,ny,time,x,y,s,w)
	end if
    
	!mean fields
    if (time.ge.Tave) then
    icount = icount + 1
    do j=0,ny
	do i=0,nx
	sa(i,j) = sa(i,j) + s(i,j)
	wa(i,j) = wa(i,j) + w(i,j)
	end do
	end do
    end if


	!snapshot data
    if(mod(k,fsnap).eq.0) then
    isnap=isnap+1
    	open(8000+isnap)
		write(8000+isnap,*)((s(i,j),i=0,nx),j=0,ny)
    	close(8000+isnap)
    	open(9000+isnap)
		write(9000+isnap,*)((w(i,j),i=0,nx),j=0,ny)
    	close(9000+isnap)
    end if
    	
 
	if(isc.eq.1.and.mod(k,100).eq.0) print*,k,tote,cfl

    
end do
call cpu_time(t2)
close(500)
close(600)
	
	!mean fields
    do j=0,ny
	do i=0,nx
	sa(i,j) = sa(i,j)/dfloat(icount)
	wa(i,j) = wa(i,j)/dfloat(icount)
	end do
	end do
    

open(7,file='cpu.txt')
write(7,*) "cpu time (with writing) = ", t2-t1
write(7,*) "cpu time (only solver)  = ", tt
close(7)


open(100,file='final.plt')
write(100,*)'variables ="x","y","s","w"'
write(100,*)'zone f=point i=',nx+1,',j=',ny+1
do j=0,ny
do i=0,nx
write(100,*) x(i),y(j),s(i,j),w(i,j)
end do
end do
close(100)

open(200,file='mean.plt')
write(200,*)'variables ="x","y","s","w"'
write(200,*)'zone f=point i=',nx+1,',j=',ny+1
do j=0,ny
do i=0,nx
write(200,*) x(i),y(j),sa(i,j),wa(i,j)
end do
end do
close(200)



end

!---------------------------------------------------------------------------!
!History
!---------------------------------------------------------------------------!
subroutine history(nx,ny,dx,dy,dt,s,cfl,tote,vm)
implicit none
integer::nx,ny
real*8 ::s(0:nx,0:ny)
integer::i,j
real*8 ::dx,dy,dt,cfl,tote,umax,vmax,vm
real*8, dimension (:), allocatable   :: a,b
real*8, dimension (:,:), allocatable :: u,v,g

allocate(u(0:nx,0:ny),v(0:nx,0:ny))

! sy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4d(a,b,dy,ny)
	do j=0,ny
	u(i,j) = b(j)
	end do
end do
deallocate(a,b)

! sx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = s(i,j)
	end do
		call c4d(a,b,dx,nx)
	do i=0,nx
	v(i,j) =-b(i)
	end do
end do
deallocate(a,b)

!compute total energy
allocate(g(0:nx,0:ny))
do i=0,nx
do j=0,ny
g(i,j)=0.5d0*(u(i,j)**2 + v(i,j)**2)
end do
end do
call simp2D(nx,ny,dx,dy,g,tote)
deallocate(g)

!compute cfl
umax=0.0d0
vmax=0.0d0
do i=0,nx
do j=0,ny
if(dabs(u(i,j)).ge.umax) umax=u(i,j)
if(dabs(v(i,j)).ge.vmax) vmax=v(i,j)
end do
end do
deallocate(u,v)
cfl=max(umax*dt/dx,vmax*dt/dy)
vm =dsqrt(umax*umax + vmax*vmax)

return
end

!---------------------------------------------------------------------------!
!Output files
!---------------------------------------------------------------------------!
subroutine outfield(nx,ny,time,x,y,s,w)
implicit none
integer::nx,ny
real*8 ::x(0:nx),y(0:ny)
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny)
integer::i,j
real*8 ::time


write(500,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',time,'"'
do j=0,ny
do i=0,nx
write(500,52) x(i),y(j),s(i,j),w(i,j)
end do
end do


51 format(a16,i8,a4,i8,a10,f12.4,a3)
52 format(4es16.6)

return
end



!---------------------------------------------------------------------------!
!TVD Runge-Kutta 3rd-order constant time step integration
!---------------------------------------------------------------------------!
subroutine tvdrk3(nx,ny,dx,dy,dt,s,w,f)
implicit none
integer::nx,ny
real*8 ::dx,dy,dt,a,b,c
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),f(0:nx,0:ny)
integer::i,j
real*8,dimension(:,:),allocatable::r,g,tb

allocate(r(0:nx,0:ny))
r = 0.0d0
allocate(g(0:nx,0:ny))
g = 0.0d0
allocate(tb(0:nx,0:ny))
tb = 0.0d0

!RHS calc
call rhs2(nx,ny,dx,dy,f,s,w,r)

!compute turbulence model
call turb_model(nx,ny,dx,dy,s,w,tb)

!first update
do j=1,ny-1
do i=1,nx-1
g(i,j) = w(i,j) + dt*(r(i,j)+tb(i,j))
end do
end do

!Elliptic Poisson solver:
call fst2(nx,ny,dx,dy,-g,s)

!compute rhs terms:
call rhs2(nx,ny,dx,dy,f,s,g,r)

!compute turbulence model
call turb_model(nx,ny,dx,dy,s,g,tb)

!second update
a = 3.0d0/4.0d0
do j=1,ny-1
do i=1,nx-1
g(i,j) = a*w(i,j) + 0.25d0*g(i,j) + 0.25d0*dt*(r(i,j)+tb(i,j))
end do
end do


!Elliptic Poisson solver:
call fst2(nx,ny,dx,dy,-g,s)

!compute rhs terms:
call rhs2(nx,ny,dx,dy,f,s,g,r)

!compute turbulence model
call turb_model(nx,ny,dx,dy,s,g,tb)

!third update
b = 1.0d0/3.0d0
c = 2.0d0/3.0d0
do j=1,ny-1
do i=1,nx-1
w(i,j) = b*w(i,j) + c*g(i,j) + c*dt*(r(i,j)+tb(i,j))
end do
end do

!Elliptic Poisson solver:
call fst2(nx,ny,dx,dy,-w,s)

deallocate(r,g,tb)

return
end


!---------------------------------------------------------------------------!
!Computing rhs of equations with 2nd-order Arakawa
!---------------------------------------------------------------------------!
subroutine rhs2(nx,ny,dx,dy,f,s,w,r)
implicit none
integer::nx,ny
real*8 ::dx,dy
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),r(0:nx,0:ny),f(0:nx,0:ny)
integer::i,j
real*8 ::Re,Ro,St,j1,j2,j3,a,b,c,d,g,h

common /phys/ Re,Ro,St

a = 1.0d0/Re
b = 1.0d0/(dx*dx)
c = 1.0d0/(dy*dy)
d = 1.0d0/(2.0d0*dx)/Ro 
h = 1.0d0/3.0d0
g = 1.0d0/(4.0d0*dx*dy)
do j=1,ny-1
do i=1,nx-1

j1 = g*((w(i+1,j)-w(i-1,j))*(s(i,j+1)-s(i,j-1)) &
       -(w(i,j+1)-w(i,j-1))*(s(i+1,j)-s(i-1,j)) )

j2 = g*(w(i+1,j)*(s(i+1,j+1)-s(i+1,j-1)) &
       -w(i-1,j)*(s(i-1,j+1)-s(i-1,j-1)) &
	   -w(i,j+1)*(s(i+1,j+1)-s(i-1,j+1)) &
	   +w(i,j-1)*(s(i+1,j-1)-s(i-1,j-1)) )

j3 = g*(w(i+1,j+1)*(s(i,j+1)-s(i+1,j)) &
       -w(i-1,j-1)*(s(i-1,j)-s(i,j-1)) &
	   -w(i-1,j+1)*(s(i,j+1)-s(i-1,j)) &
	   +w(i+1,j-1)*(s(i+1,j)-s(i,j-1)) )

r(i,j) = f(i,j) &	!Forcing
       - (j1+j2+j3)*h & !Jacobian
	   + (s(i+1,j)-s(i-1,j))*d  & !Coriolis
	   - St*w(i,j) & !Ekman
       + a*((w(i+1,j)-2.0d0*w(i,j)+w(i-1,j))*b  & !Viscous
           +(w(i,j+1)-2.0d0*w(i,j)+w(i,j-1))*c ) 
 

end do
end do


return
end

!---------------------------------------------------------------------------!
!Turbulence modeling section for QG
!---------------------------------------------------------------------------!

!---------------------------------------------------------------------------!
!Routine for different turbulence model control
!---------------------------------------------------------------------------!
subroutine turb_model(nx,ny,dx,dy,s,w,tb)
	implicit none
	integer :: nx, ny, iturb
	real*8 :: dx, dy
	real*8, dimension(0:nx,0:ny) :: s,w,tb

	common /turb/ iturb

	if (iturb.eq.1) then
		call dynamic_smagorinsky(nx,ny,dx,dy,s,w,tb)
	else if (iturb.eq.2) then
		call dynamic_leith(nx,ny,dx,dy,s,w,tb)
	else if (iturb.eq.3) then
		call approximate_deconvolution(nx,ny,dx,dy,s,w,tb)
	else if (iturb.eq.4) then
		call standard_smagorinsky(nx,ny,dx,dy,s,w,tb)
	else if (iturb.eq.5) then
		call standard_leith(nx,ny,dx,dy,s,w,tb)
	end if
	return
end

!---------------------------------------------------------------------------!
!Standard Smagorinsky subroutine - vorticity streamfunction formulation
!---------------------------------------------------------------------------!
subroutine standard_smagorinsky(nx,ny,dx,dy,s,w,tb)
	implicit none
	integer :: nx, ny, i, j
	real*8, dimension(0:nx,0:ny) :: s,w,tb
	real*8, dimension(:,:),allocatable :: st,lap
	real*8 :: dx, dy, cs, del, d2wdy2, d2wdx2

	cs = 0.2d0
	del = dsqrt(dx*dy)

	allocate(st(0:nx,0:ny))
	st = 0.0d0

	call strain(w,s,st,nx,ny,dx,dy)

	allocate(lap(0:nx,0:ny))
	lap=0.0d0
	!Compute laplacian
	do j = 1,ny-1
		do i = 1,nx-1
			d2wdy2 = (w(i, j+1) + w(i, j-1) - 2.0 * w(i, j)) / (dy * dy)
			d2wdx2 = (w(i+1, j) + w(i-1, j) - 2.0 * w(i, j)) / (dx * dx)
			lap(i,j) = d2wdx2 + d2wdy2
		end do
	end do

	do j=1,ny-1
	do i=1,nx-1
		tb(i,j)=cs*cs*del*del*st(i,j)*lap(i,j)
	end do
	end do
	
	deallocate(st,lap)

	return
end


!---------------------------------------------------------------------------!
!Dynamic Smagorinsky subroutine - vorticity streamfunction formulation
!---------------------------------------------------------------------------!
subroutine dynamic_smagorinsky(nx,ny,dx,dy,s,w,tb)
	implicit none
	integer :: nx, ny
	real*8 :: dx, dy
	real*8, dimension(0:nx,0:ny) :: s,w,tb

	integer :: i,j
	double precision :: kappa2,dd,nn,csd
	double precision, dimension (:,:), allocatable :: w_f,s_f,fjc,jcf,st,lwf,wlap,jc
	double precision :: d2wdx2, d2wdy2

	kappa2 = 2.0d0

	!compute jacobian of filtered variables
	allocate(wlap(0:nx,0:ny))
	wlap = 0.0d0

	!Compute laplacian
	do j = 1,ny-1
		do i = 1,nx-1
			d2wdy2 = (w(i, j+1) + w(i, j-1) - 2.0 * w(i, j)) / (dy * dy)
			d2wdx2 = (w(i+1, j) + w(i-1, j) - 2.0 * w(i, j)) / (dx * dx)
			wlap(i,j) = d2wdx2 + d2wdy2
		end do
	end do


	allocate(w_f(0:nx,0:ny))
	w_f = 0.0d0

	allocate(s_f(0:nx,0:ny))
	s_f = 0.0d0

	allocate(fjc(0:nx,0:ny))
	fjc = 0.0d0
	
	allocate(jc(0:nx,0:ny))
	jc = 0.0d0


	call filter(nx,ny,w,w_f)
	call filter(nx,ny,s,s_f)

	call jacobian_2(nx,ny,dx,dy,w,s,jc)
	call jacobian_2(nx,ny,dx,dy,w_f,s_f,fjc)

	!compute filtered jacobian
	allocate(jcf(0:nx,0:ny))
	jcf = 0.0d0

	call filter(nx,ny,jc,jcf)

	!compute laplacian of wf 
	allocate(lwf(0:nx,0:ny))
	lwf = 0.0d0

	do j = 1,ny-1
		do i = 1,nx-1
			d2wdy2 = (w_f(i, j+1) + w_f(i, j-1) - 2.0 * w_f(i, j)) / (dy * dy)
			d2wdx2 = (w_f(i+1, j) + w_f(i-1, j) - 2.0 * w_f(i, j)) / (dx * dx)
			lwf(i,j) = d2wdx2 + d2wdy2
		end do
	end do
	

	!compute strain
	allocate(st(0:nx,0:ny))
	st = 0.0d0

	call strain(w,s,st,nx,ny,dx,dy)
	
	!get filtered st ==> sf
    call filter(nx,ny,st,s_f)

	!compute psi_f L on test filter ==> lwf
	do j=0,ny
	do i=0,nx
	lwf(i,j)=s_f(i,j)*lwf(i,j)
	end do
	end do

	!compute |S|L ==> wf on grid filter
	do j=0,ny
	do i=0,nx
	w_f(i,j)=st(i,j)*wlap(i,j)
	end do
	end do

	!compute  filtered |S|L on grid filter
	call filter(nx,ny,w_f,s_f)

	nn = 0.0d0
	dd = 0.0d0
	!compute (cs*delta)^2 =csd
	do j=1,ny-1
	do i=1,nx-1
	nn = nn + (fjc(i,j) - jcf(i,j))*(kappa2*lwf(i,j) - s_f(i,j))
	dd = dd + (kappa2*lwf(i,j) - s_f(i,j))*(kappa2*lwf(i,j) - s_f(i,j))
	end do
	end do
	
	!compute csd
	csd = dabs(nn/(dd+1.0d-10))

	!Final source term
	do j=1,ny-1
	do i=1,nx-1
		tb(i,j) = csd*st(i,j)*wlap(i,j)
	end do
	end do

	deallocate(w_f,s_f,fjc,jcf,st,lwf,wlap,jc)

	return
end

!-----------------------------------------------------------------!
!Filter
!-----------------------------------------------------------------!
subroutine filter(nx,ny,w,wf)
implicit none
integer ::nx,ny,ifil,i,j
double precision::w(0:nx,0:ny),wf(0:nx,0:ny),dd

dd=1.0d0/16.0d0

do j=1,ny-1
do i=1,nx-1
wf(i,j) = dd*(4.0d0*w(i,j) &
       + 2.0d0*(w(i+1,j) + w(i-1,j) + w(i,j+1) + w(i,j-1)) &
	   + w(i+1,j+1) + w(i-1,j-1) + w(i+1,j-1) + w(i-1,j+1))
end do
end do

return
end


!---------------------------------------------------------------------------!
!Computing 2nd-order Arakawa based Jacobian
!---------------------------------------------------------------------------!
subroutine jacobian_2(nx,ny,dx,dy,w,s,jc)
implicit none
integer::nx,ny
real*8 ::dx,dy
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),jc(0:nx,0:ny)
integer::i,j
real*8 ::Re,Ro,St,j1,j2,j3,a,b,c,d,g,h

common /phys/ Re,Ro,St

a = 1.0d0/Re
b = 1.0d0/(dx*dx)
c = 1.0d0/(dy*dy)
d = 1.0d0/(2.0d0*dx)/Ro 
h = 1.0d0/3.0d0
g = 1.0d0/(4.0d0*dx*dy)

do j=1,ny-1
do i=1,nx-1

j1 = g*((w(i+1,j)-w(i-1,j))*(s(i,j+1)-s(i,j-1)) &
       -(w(i,j+1)-w(i,j-1))*(s(i+1,j)-s(i-1,j)) )

j2 = g*(w(i+1,j)*(s(i+1,j+1)-s(i+1,j-1)) &
       -w(i-1,j)*(s(i-1,j+1)-s(i-1,j-1)) &
	   -w(i,j+1)*(s(i+1,j+1)-s(i-1,j+1)) &
	   +w(i,j-1)*(s(i+1,j-1)-s(i-1,j-1)) )

j3 = g*(w(i+1,j+1)*(s(i,j+1)-s(i+1,j)) &
       -w(i-1,j-1)*(s(i-1,j)-s(i,j-1)) &
	   -w(i-1,j+1)*(s(i,j+1)-s(i-1,j)) &
	   +w(i+1,j-1)*(s(i+1,j)-s(i,j-1)) )

jc(i,j) = (j1+j2+j3)*h !Jacobian

end do
end do

return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for smagorinsky kernel calculation
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine strain(w,s,st,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer:: nx,ny,i,j
double precision,dimension(0:nx,0:ny) :: w, s
double precision :: dx, dy
double precision :: st(0:nx,0:ny)
double precision, dimension(:,:), allocatable :: dsdy, d2sdx, d2sdy, d2sdxdy

!Calculating Smag Turbulence model invariants
allocate(d2sdxdy(0:nx,0:ny))
d2sdxdy = 0.0d0

allocate(d2sdx(0:nx,0:ny))
d2sdx = 0.0d0

allocate(d2sdy(0:nx,0:ny))
d2sdy = 0.0d0

allocate(dsdy(0:nx,0:ny))
dsdy = 0.0d0

do j = 1,ny-1
  do i = 1,nx-1
    dsdy(i,j) = (s(i,j+1)-s(i,j-1))/(2.0d0*dy)
    d2sdx(i,j) = (s(i+1,j)+s(i-1,j)-2.0d0*s(i,j))/(dx*dx)
    d2sdy(i,j) = (s(i,j+1)+s(i,j-1)-2.0d0*s(i,j))/(dy*dy)
  end do
end do

do j = 1,ny-1
  do i = 1,nx-1
    d2sdxdy(i,j) = (dsdy(i+1,j)-dsdy(i-1,j))/(2.0d0*dx)
  end do
end do

!Smag invariant
do j = 1,ny-1
  do i = 1,nx-1
    st(i,j) = dsqrt(4.0d0*d2sdxdy(i,j)**2 + (d2sdx(i,j)-d2sdy(i,j))**2)
  end do
end do

deallocate(d2sdxdy,dsdy,d2sdx,d2sdy)

return
end

!---------------------------------------------------------------------------!
!Standard Leith subroutine - vorticity streamfunction formulation
!---------------------------------------------------------------------------!
subroutine standard_leith(nx,ny,dx,dy,s,w,tb)
	implicit none
	integer :: nx, ny, i, j
	real*8, dimension(0:nx,0:ny) :: s,w,tb
	real*8, dimension(:,:),allocatable :: st,lap
	real*8 :: dx, dy, cs, del, d2wdy2, d2wdx2

	cs = 0.2d0
	del = dsqrt(dx*dy)

	allocate(st(0:nx,0:ny))
	st = 0.0d0

	call vort_grad(w,st,nx,ny,dx,dy)

	allocate(lap(0:nx,0:ny))
	lap=0.0d0
	!Compute laplacian
	do j = 1,ny-1
		do i = 1,nx-1
			d2wdy2 = (w(i, j+1) + w(i, j-1) - 2.0 * w(i, j)) / (dy * dy)
			d2wdx2 = (w(i+1, j) + w(i-1, j) - 2.0 * w(i, j)) / (dx * dx)
			lap(i,j) = d2wdx2 + d2wdy2
		end do
	end do

	do j=1,ny-1
	do i=1,nx-1
		tb(i,j)=cs*cs*cs*del*del*del*st(i,j)*lap(i,j)
	end do
	end do
	
	deallocate(st,lap)

	return
end


!---------------------------------------------------------------------------!
!Dynamic Leith subroutine - vorticity streamfunction formulation
!---------------------------------------------------------------------------!
subroutine dynamic_leith(nx,ny,dx,dy,s,w,tb)
	implicit none

	integer :: nx, ny
	real*8 :: dx, dy
	real*8, dimension(0:nx,0:ny) :: s,w,tb

	integer :: i,j
	double precision :: kappa2,dd,nn,csd
	double precision, dimension (:,:), allocatable :: w_f,s_f,fjc,jcf,st,lwf,wlap,jc
	double precision :: d2wdx2, d2wdy2

	kappa2 = 2.0d0

	!compute jacobian of filtered variables
	allocate(wlap(0:nx,0:ny))
	wlap = 0.0d0

	!Compute laplacian
	do j = 1,ny-1
		do i = 1,nx-1
			d2wdy2 = (w(i, j+1) + w(i, j-1) - 2.0 * w(i, j)) / (dy * dy)
			d2wdx2 = (w(i+1, j) + w(i-1, j) - 2.0 * w(i, j)) / (dx * dx)
			wlap(i,j) = d2wdx2 + d2wdy2
		end do
	end do


	allocate(w_f(0:nx,0:ny))
	w_f = 0.0d0

	allocate(s_f(0:nx,0:ny))
	s_f = 0.0d0

	allocate(fjc(0:nx,0:ny))
	fjc = 0.0d0
	
	allocate(jc(0:nx,0:ny))
	jc = 0.0d0


	call filter(nx,ny,w,w_f)
	call filter(nx,ny,s,s_f)

	call jacobian_2(nx,ny,dx,dy,w,s,jc)
	call jacobian_2(nx,ny,dx,dy,w_f,s_f,fjc)

	!compute filtered jacobian
	allocate(jcf(0:nx,0:ny))
	jcf = 0.0d0

	call filter(nx,ny,jc,jcf)

	!compute laplacian of wf 
	allocate(lwf(0:nx,0:ny))
	lwf = 0.0d0

	do j = 1,ny-1
		do i = 1,nx-1
			d2wdy2 = (w_f(i, j+1) + w_f(i, j-1) - 2.0 * w_f(i, j)) / (dy * dy)
			d2wdx2 = (w_f(i+1, j) + w_f(i-1, j) - 2.0 * w_f(i, j)) / (dx * dx)
			lwf(i,j) = d2wdx2 + d2wdy2
		end do
	end do
	

	!compute strain
	allocate(st(0:nx,0:ny))
	st = 0.0d0

	call vort_grad(w,st,nx,ny,dx,dy)
	
	!get filtered st ==> sf
    call filter(nx,ny,st,s_f)

	!compute psi_f L on test filter ==> lwf
	do j=0,ny
	do i=0,nx
	lwf(i,j)=s_f(i,j)*lwf(i,j)
	end do
	end do

	!compute |S|L ==> wf on grid filter
	do j=0,ny
	do i=0,nx
	w_f(i,j)=st(i,j)*wlap(i,j)
	end do
	end do

	!compute  filtered |S|L on grid filter
	call filter(nx,ny,w_f,s_f)

	nn = 0.0d0
	dd = 0.0d0
	!compute (cs*delta)^2 =csd
	do j=0,ny
	do i=0,nx 
	nn = nn + (fjc(i,j) - jcf(i,j))*(kappa2*lwf(i,j) - s_f(i,j))
	dd = dd + (kappa2*lwf(i,j) - s_f(i,j))*(kappa2*lwf(i,j) - s_f(i,j))
	end do
	end do
	
	!compute csd
	csd = dabs(nn/(dd+1.0d-10))


	!Final source term
	do j=1,ny-1
	do i=1,nx-1
		tb(i,j) = csd*st(i,j)*wlap(i,j)
	end do
	end do

	deallocate(w_f,s_f,fjc,jcf,st,lwf,wlap,jc)

	




	return
end


!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
!Subroutine for Leith kernel calculation
!Validated
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
subroutine vort_grad(w,st,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer :: i,j,nx,ny
double precision,dimension(0:nx,0:ny) :: w, s
double precision :: dx, dy
double precision :: st(0:nx,0:ny)
double precision, dimension(:,:), allocatable :: dwdx,dwdy

!Calculating Leith turbulence model invariants
allocate(dwdy(0:nx,0:ny))
dwdy = 0.0d0
allocate(dwdx(0:nx,0:ny))
dwdx = 0.0d0

do j = 1,ny-1
do i = 1,nx-1
dwdy(i,j) = (w(i,j+1)-w(i,j-1))/(2.0d0*dy)
dwdx(i,j) = (w(i+1,j)-w(i-1,j))/(2.0d0*dx)
end do
end do

!Leith invariant
do j = 1,ny-1
  do i = 1,nx-1
    st(i,j) = dsqrt(dwdx(i,j)**2 + dwdy(i,j)**2)
  end do
end do

deallocate(dwdx,dwdy)

return
end

!---------------------------------------------------------------------------!
!Approximate deconvolution subroutine - vorticity streamfunction formulation
!---------------------------------------------------------------------------!
subroutine approximate_deconvolution(nx,ny,dx,dy,s,w,tb)
	implicit none
	integer :: nx, ny, i, j
	real*8 :: dx, dy
	real*8, dimension(0:nx,0:ny) :: s,w,tb
	real*8, dimension(:,:),allocatable :: ss,ws,jc,jcad,jcadf


	!Allocate storage for deconvolution
	allocate(ss(0:nx,0:ny))
	ss = 0.0d0

	allocate(ws(0:nx,0:ny))
	ws = 0.0d0

	call adm(nx,ny,w,ws)
	call adm(nx,ny,s,ss)

	allocate(jcad(0:nx,0:ny))
	jcad = 0.0d0
	!Calculate ad jacobian
	call jacobian_2(nx,ny,dx,dy,ws,ss,jcad)

	!Filter
	allocate(jcadf(0:nx,0:ny))
	jcadf = 0.0d0

	call filter(nx,ny,jcad,jcadf)

	!Call jacobian of filtered field
	allocate(jc(0:nx,0:ny))
	jc = 0.0d0
	call jacobian_2(nx,ny,dx,dy,w,s,jc)

	do j = 1,ny-1
		do i = 1,nx-1
			tb(i,j) = jc(i,j)-jcadf(i,j)
		end do
	end do

	deallocate(ws,ss,jc,jcad,jcadf)

	return
end


!---------------------------------------------------------------------------!
!Iterative deconvolution - Van Cittert iterations
!Trapezoidal filter
!---------------------------------------------------------------------------!
subroutine adm(nx,ny,w,ws)
	implicit none
	integer :: nx, ny, nadm, k, i, j
	double precision, dimension(0:nx,0:ny) :: w,ws
	double precision, dimension(:,:),allocatable :: wg

	nadm = 3

	allocate(wg(0:nx,0:ny))
	wg = 0.0d0

	do k = 1, nadm    
    	!compute filtered value of guess
    	call filter(nx,ny,w,wg)  
		do j=1,ny-1
			do i=1,nx-1
				ws(i,j) = w(i,j) + (w(i,j) - wg(i,j))
			end do
	    end do
	end do

	deallocate(wg)

	return
end

!---------------------------------------------------------------------------!
!Routines for fast Poisson solver
!---------------------------------------------------------------------------!

!---------------------------------------------------------------------------!
!fast sin transformation direct poisson solver
!fast direct poisson solver for homogeneous drichlet boundary conditions
!using discreate fast sin transformation along x and y axis 
!second order formula
!---------------------------------------------------------------------------!
subroutine fst2(nx,ny,dx,dy,f,u) 
implicit none
integer::i,j,nx,ny,isign
real*8,dimension(0:nx,0:ny)::u,f
real*8,dimension(:,:),allocatable:: ft
real*8::dx,dy,pi,alpha

pi=4.0d0*datan(1.0d0)

allocate(ft(0:nx,0:ny))

!rename for rhs (not to replace the data)
do i=0,nx
do j=0,ny
ft(i,j) = f(i,j)
end do
end do

!fast inverse fourier sine transform of source term:
isign=-1
call sinft2d(nx,ny,isign,ft)

!Compute fourier coefficient of u:
do i=1,nx-1
do j=1,ny-1
alpha=2.0d0/(dx*dx)*(dcos(pi*dfloat(i)/dfloat(nx))-1.0d0) &
     +2.0d0/(dy*dy)*(dcos(pi*dfloat(j)/dfloat(ny))-1.0d0)

u(i,j)=ft(i,j)/alpha

end do
end do

!fast forward fourier sine transform:
isign=1
call sinft2d(nx,ny,isign,u)

deallocate(ft)

return
end

!---------------------------------------------------------------------------!
!Compute fast fourier sine transform for 2D data
!Homogeneous Drichlet Boundary Conditios (zero all boundaries)
!Input:: u(0:nx,0:ny) 
!        where indices 0,nx,ny represent boundary data and should be zero
!Output::override
!Automatically normalized
!isign=-1 is inverse transform and 2/N is already applied 
!        (from grid data to fourier coefficient)
!isign=+1 is forward transform
!        (from fourier coefficient to grid data)
!---------------------------------------------------------------------------!
subroutine sinft2d(nx,ny,isign,u)
implicit none
integer::nx,ny,isign
real*8 ::u(0:nx,0:ny)
integer::i,j
real*8, dimension(:),allocatable  :: v

if (isign.eq.-1) then !inverse transform
! compute inverse sine transform to find fourier coefficients of f in x-direction
allocate(v(nx))
do j=1,ny-1
	do i=1,nx
	v(i) = u(i-1,j)
	end do
	call sinft(v,nx)
	do i=2,nx
	u(i-1,j)=v(i)*2.0d0/dfloat(nx)
	end do
end do
deallocate(v)
allocate(v(ny))
! compute inverse sine transform to find fourier coefficients of f in y-direction
do i=1,nx-1
	do j=1,ny
	v(j) = u(i,j-1)
	end do
	call sinft(v,ny)
	do j=2,ny
	u(i,j-1)=v(j)*2.0d0/dfloat(ny)
	end do
end do
deallocate(v)
else  !forward transform
! compute forward sine transform to find fourier coefficients of f in x-direction
allocate(v(nx))
do j=1,ny-1
	do i=1,nx
	v(i) = u(i-1,j)
	end do
	call sinft(v,nx)
	do i=2,nx
	u(i-1,j)=v(i)
	end do
end do
deallocate(v)
allocate(v(ny))
! compute forward sine transform to find fourier coefficients of f in y-direction
do i=1,nx-1
	do j=1,ny
	v(j) = u(i,j-1)
	end do
	call sinft(v,ny)
	do j=2,ny
	u(i,j-1)=v(j)
	end do
end do
deallocate(v)
end if

return
end 

!---------------------------------------------------------------------------!
!calculates sine transform of a set of n real valued data points, y(1,2,..n)
!y(1) is zero, y(n) not need to be zero, but y(n+1)=0
!also calculates inverse transform, but output should be multiplied by 2/n
!n should be powers of 2
!use four1 and realft routines
!---------------------------------------------------------------------------!
subroutine sinft(y,n)
implicit none
integer::n,j,m
real*8 ::wr,wi,wpr,wpi,wtemp,theta,y1,y2,sum
real*8 ::y(n)
      theta=3.14159265358979d0/dble(n)
      wr=1.0d0
      wi=0.0d0
      wpr=-2.0d0*dsin(0.5d0*theta)**2
      wpi=dsin(theta)
      y(1)=0.0
      m=n/2
      do 11 j=1,m
        wtemp=wr
        wr=wr*wpr-wi*wpi+wr
        wi=wi*wpr+wtemp*wpi+wi
        y1=wi*(y(j+1)+y(n-j+1))
        y2=0.5*(y(j+1)-y(n-j+1))
        y(j+1)=y1+y2
        y(n-j+1)=y1-y2
11    continue
      call realft(y,m,+1)
      sum=0.0
      y(1)=0.5*y(1)
      y(2)=0.0
      do 12 j=1,n-1,2
        sum=sum+y(j)
        y(j)=y(j+1)
        y(j+1)=sum
12    continue
return
end

!---------------------------------------------------------------------------!
!computes real fft
!---------------------------------------------------------------------------!
subroutine realft(data,n,isign)
implicit none
integer::n,isign,i,i1,i2,i3,i4,n2p3
real*8 ::wr,wi,wpr,wpi,wtemp,theta,c1,c2,h2r,h2i,h1r,h1i
real*8 ::data(*)
real   ::wrs,wis 
      theta=6.28318530717959d0/2.0d0/dble(n)
      c1=0.5
      if (isign.eq.1) then
        c2=-0.5
        call four1(data,n,+1)
      else
        c2=0.5
        theta=-theta
      endif
      wpr=-2.0d0*dsin(0.5d0*theta)**2
      wpi=dsin(theta)
      wr=1.0d0+wpr
      wi=wpi
      n2p3=2*n+3
      do 11 i=2,n/2+1
        i1=2*i-1
        i2=i1+1
        i3=n2p3-i2
        i4=i3+1
        wrs=sngl(wr)
        wis=sngl(wi)
        h1r=c1*(data(i1)+data(i3))
        h1i=c1*(data(i2)-data(i4))
        h2r=-c2*(data(i2)+data(i4))
        h2i=c2*(data(i1)-data(i3))
        data(i1)=h1r+wrs*h2r-wis*h2i
        data(i2)=h1i+wrs*h2i+wis*h2r
        data(i3)=h1r-wrs*h2r+wis*h2i
        data(i4)=-h1i+wrs*h2i+wis*h2r
        wtemp=wr
        wr=wr*wpr-wi*wpi+wr
        wi=wi*wpr+wtemp*wpi+wi
11    continue
      if (isign.eq.1) then
        h1r=data(1)
        data(1)=h1r+data(2)
        data(2)=h1r-data(2)
      else
        h1r=data(1)
        data(1)=c1*(h1r+data(2))
        data(2)=c1*(h1r-data(2))
        call four1(data,n,-1)
      endif
return
end

!---------------------------------------------------------------------------!
!FFT routine for 1-dimensional data 
!---------------------------------------------------------------------------!
subroutine four1(data,nn,isign)
implicit none
integer:: nn,isign,i,j,m,n,mmax,istep
real*8 :: wr,wi,wpr,wpi,wtemp,theta,tempr,tempi
real*8 :: data(*)
      n=2*nn
      j=1
      do 11 i=1,n,2
        if(j.gt.i)then
          tempr=data(j)
          tempi=data(j+1)
          data(j)=data(i)
          data(j+1)=data(i+1)
          data(i)=tempr
          data(i+1)=tempi
        endif
        m=n/2
1       if ((m.ge.2).and.(j.gt.m)) then
          j=j-m
          m=m/2
        go to 1
        endif
        j=j+m
11    continue
      mmax=2
2     if (n.gt.mmax) then
        istep=2*mmax
        theta=6.28318530717959d0/(isign*mmax)
        wpr=-2.d0*dsin(0.5d0*theta)**2
        wpi=dsin(theta)
        wr=1.d0
        wi=0.d0
        do 13 m=1,mmax,2
          do 12 i=m,n,istep
            j=i+mmax
            tempr=sngl(wr)*data(j)-sngl(wi)*data(j+1)
            tempi=sngl(wr)*data(j+1)+sngl(wi)*data(j)
            data(j)=data(i)-tempr
            data(j+1)=data(i+1)-tempi
            data(i)=data(i)+tempr
            data(i+1)=data(i+1)+tempi
12        continue
          wtemp=wr
          wr=wr*wpr-wi*wpi+wr
          wi=wi*wpr+wtemp*wpi+wi
13      continue
        mmax=istep
      go to 2
      endif
return
end


!---------------------------------------------------------------------------!
!Routines for compact schemes
!---------------------------------------------------------------------------!

!-------------------------------------------------!
!solution tridiagonal systems 
!-------------------------------------------------!

subroutine tdma(a,b,c,r,x,s,e)
implicit none
integer s,e,i
real*8, dimension(s:e) ::a,b,c,r,x    

! forward elimination phase
do i=s+1,e
b(i) = b(i) - a(i)/b(i-1)*c(i-1)
r(i) = r(i) - a(i)/b(i-1)*r(i-1)
end do
! backward substitution phase 
x(e) = r(e)/b(e)
do i=e-1,s,-1
x(i) = (r(i)-c(i)*x(i+1))/b(i)
end do
return
end

!------------------------------------------------------------------!
!c4d: 	4th-order compact scheme for the first order derivative(up)
!		drichlet boundary condition u(0)=given, u(n)=given
!		3-4-3
!		tested
!------------------------------------------------------------------!
subroutine c4d(u,up,h,n)
implicit none
integer :: n,i
real*8  :: h
real*8, dimension (0:n)	:: u,a,b,c,r,up

i=0
b(i) = 1.0d0
c(i) = 2.0d0
r(i) = (-5.0d0*u(i) + 4.0d0*u(i+1) + u(i+2))/(2.0d0*h)

do i=1,n-1
a(i) = 1.0d0/4.0d0
b(i) = 1.0d0
c(i) = 1.0d0/4.0d0
r(i) = 3.0d0/2.0d0*(u(i+1)-u(i-1))/(2.0d0*h)
end do

i=n
a(i) = 2.0d0
b(i) = 1.0d0
r(i) = (-5.0d0*u(i) + 4.0d0*u(i-1) + u(i-2))/(-2.0d0*h)

call tdma(a,b,c,r,up,0,n)

return
end

!------------------------------------------------------------------!
!c4dd:  4th-order compact scheme for the second order derivative(upp)
!		drichlet boundary condition u(0)=given, u(n)=given
!		3-4-3 sided derivative formula
!		tested 
!----------------------------------------------------!
subroutine c4dd(u,upp,h,n)
implicit none
integer :: n,i
real*8  :: h
real*8, dimension (0:n)	:: u,a,b,c,r,upp

i=0
b(i) = 1.0d0
c(i) = 11.0d0
r(i) = (13.0d0*u(i)-27.0d0*u(i+1)+15.0d0*u(i+2)-u(i+3))/(h*h)

do i=1,n-1
a(i) = 1.0d0/10.0d0
b(i) = 1.0d0
c(i) = 1.0d0/10.0d0
r(i) = 6.0d0/5.0d0*(u(i-1)-2.0d0*u(i)+u(i+1))/(h*h) 
end do

i=n
a(i) = 11.0d0
b(i) = 1.0d0
r(i) = (13.0d0*u(i)-27.0d0*u(i-1)+15.0d0*u(i-2)-u(i-3))/(h*h)


call tdma(a,b,c,r,upp,0,n)

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