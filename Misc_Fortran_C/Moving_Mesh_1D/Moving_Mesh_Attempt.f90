!Program for moving mesh in 1D FDM
!Romit Maulik
!Refer introductory chapter of Adaptive Moving Mesh Methods - Springer
!Burgers equation with Dirichlet BCs

program mm1d
implicit none

integer :: nx,i,t,nt,isol,iform
real*8,dimension(:),allocatable :: u,x,dxdt,utemp
real*8 :: dt,xl,dx,pi,x0

common/nlform/iform

pi = 4.0d0*datan(1.0d0)
dt = 1d-3
nt = nint(1d3)
nx = 41
xl = 2.0d0*pi
dx = xl/dfloat(nx)
isol = 2			!Euler FWD or RK3
iform = 1			![1] - Central or [2] - QUICK for Nonlinear term

allocate(u(1:nx))  !Not periodic BC
allocate(x(1:nx))

!Setting up initial condition
x0 = -pi
do i = 1,nx
  x(i) = x0+dx
  u(i) = -dsin(x(i))
  x0 = x0+dx
end do

open(8,file='Initial_Plot.plt')
write(8,*) 'variables ="x","f(x)"'
write(8,*)'zone f=point i=',nx,''
	do i=1,nx
	write(8,*) x(i),u(i)
	end do
close(8)

!Calculating mesh speed - dxdt
allocate(dxdt(1:nx))
allocate(utemp(1:nx))

if (isol==1) then !forward Euler

open(8,file='MM_Plot.plt')
do t = 1,nt

call meshspeed(u,x,dxdt,nx)

call vel_update(u,x,dxdt,nx,utemp)

!New mesh distribution
do i = 1,nx
  x(i) = x(i)+dxdt(i)*dt
end do

!New velocity distribution
do i = 1,nx
  u(i) = u(i)+utemp(i)*dt
end do


!if (mod(t,nt/10)==0) then

write(8,*)'variables ="xnew","f(x)"'
write(8,*)'zone f=point i=',nx,',t="case',t,'"'
	do i=1,nx
	write(8,*) x(i),u(i)
	end do

!end if

end do

close(8)

else if (isol==2) then!Rk3

  
open(8,file='MM_Plot.plt')
call rk3(u,x,dxdt,nx,nt,dt)
close(8)



end if



end



!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
!subroutine for rk3 time integration
!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
subroutine rk3(u,x,dxdt,nx,nt,dt)
implicit none

integer :: nx,t,nt,i
real*8 :: dt
real*8,dimension(1:nx) :: u,x,dxdt
real*8,dimension(1:nx) :: u1,u2,utemp,x1,x2

do i = 1,nx
u1(i) = u(i)
u2(i) = u(i)
utemp(i) = u(i)
x1(i) = x(i)
x2(i) = x(i)
end do


do t = 1,nt

	!Stage 1
	call meshspeed(u,x,dxdt,nx)

    call vel_update(u,x,dxdt,nx,utemp)
	
	do i = 1,nx
    u1(i) = u(i) + dt*utemp(i)
	end do

    
	!New stage mesh distribution
    do i = 1,nx
  	x1(i) = x(i)+dxdt(i)*dt
    end do

	!Stage 2
	call meshspeed(u1,x1,dxdt,nx)

    call vel_update(u1,x1,dxdt,nx,utemp)

	do i = 1,nx
    u2(i) = 0.75d0*u(i) + 0.25d0*u1(i) + 0.25*utemp(i)*dt 
    
	x2(i) = 0.75d0*x(i) + 0.25d0*x1(i) + 0.25*dt*dxdt(i)
    end do

    !stage 3

	call meshspeed(u2,x2,dxdt,nx)

    call vel_update(u2,x2,dxdt,nx,utemp)
	
	do i = 1,nx
    u(i) = 1.0d0/3.0d0*u(i) + 2.0d0/3.0d0*u2(i) + 2.0d0/3.0d0*utemp(i)*dt 
    
    x(i) = 1.0d0/3.0d0*x(i) + 2.0d0/3.0d0*x2(i) + 2.0d0/3.0d0*dxdt(i)*dt
    end do


!if (mod(t,nt/10)==0) then

write(8,*)'variables ="xnew","f(x)"'
write(8,*)'zone f=point i=',nx,',t="case',t,'"'
	do i=1,nx
	write(8,*) x(i),u(i)
	end do

!end if


end do











return
end





!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
!subroutine to calculate meshspeed call meshspeed(u,x,dxdt,nx)
!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
subroutine meshspeed(u,x,dxdt,nx)
implicit none

integer :: i,j,nx
real*8,dimension(1:nx) :: dxdt,u,x
real*8,dimension(:),allocatable :: uxx,rho
real*8 :: alphah,sum,tau,dxi


allocate(uxx(1:nx))

uxx(1) 	= 	2.0d0*((x(2)-x(1))*(u(3)-u(1))-(x(3)-x(1))*(u(2)-u(1)))/((x(3)-x(1))*(x(2)-x(1))*(x(3)-x(2)))
uxx(nx) = 	2.0d0*((x(nx-1)-x(nx))*(u(nx-2)-u(nx))-(x(nx-2)-x(nx))*(u(nx-1)-u(nx)))&
			&/((x(nx-2)-x(nx))*(x(nx-1)-x(nx))*(x(nx-2)-x(nx-1)))

do i = 2,nx-1
  uxx(i) = 2.0d0/(x(i+1)-x(i-1))*((u(i+1)-u(i))/(x(i+1)-x(i))-(u(i)-u(i-1))/(x(i)-x(i-1)))
end do

sum = 0.0d0

do j = 2,nx
    sum = sum + (0.5d0*(x(j)-x(j-1))*(dabs(uxx(j))**(2.0d0/3.0d0)+dabs(uxx(j-1))**(2.0d0/3.0d0)))**3 
end do

alphah = max(1,sum)

allocate(rho(1:nx))

do i = 1,nx
  rho(i) = (1+1/alphah*dabs(uxx(i))**2)**(1.0d0/3.0d0)
end do

!Smoothing rho

do i = 2,nx-1
  rho(i) = 0.25d0*rho(i-1)+0.5d0*rho(i)+0.25d0*rho(i+1)
end do

rho(1) = 0.5d0*rho(1)+0.5d0*rho(2)
rho(nx) = 0.5d0*rho(nx-1)+0.5d0*rho(nx)

dxdt(1) = 0.0d0
dxdt(nx) = 0.0d0

tau = 0.01d0
dxi = 1.0d0/dfloat(nx-1)

do i = 2,nx-1
  dxdt(i) = 1.0d0/(rho(i)*tau*dxi**2)*((rho(i+1)+rho(i))/2.0d0*(x(i+1)-x(i))-(rho(i)+rho(i-1))/2.0d0*(x(i)-x(i-1)))
end do

deallocate(rho)

return
end




!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
!subroutine to calculate new velocty
!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
subroutine vel_update(u,x,dxdt,nx,v)
implicit none

integer :: nx,i,iform
real*8,dimension(1:nx) :: u,x,dxdt,v
real*8 :: eps,ta,tb,tc,td
real*8,dimension(:),allocatable :: rhs,nl

common/nlform/iform

!eps = 0.0001d0
eps = 0.0d0

allocate(rhs(1:nx))
allocate(nl(1:nx))

if (iform==1) then

do i = 2,nx-1
  ta = (u(i+1)-u(i))/(x(i+1)-x(i))
  tb = (u(i)-u(i-1))/(x(i)-x(i-1)) 
  tc = 0.5d0*(u(i+1)**2-u(i-1)**2)/(x(i+1)-x(i-1))
  rhs(i) = 2*eps/(x(i+1)-x(i-1))*(ta-tb)-tc
end do

do i = 2,nx-1
  nl(i) = -(u(i+1)-u(i-1))/(x(i+1)-x(i-1))*dxdt(i)
end do


do i = 2,nx-1
  v(i) = rhs(i)- nl(i)
end do

v(1) = 0.0d0
v(nx) = 0.0d0

else if (iform==2) then

do i = 2,nx-1
	if (u(i)>0.0d0) then
		ta = (u(i+1)-u(i))/(x(i+1)-x(i))
		tb = (u(i)-u(i-1))/(x(i)-x(i-1))
        td = (0.75d0*u(i-1)+3.0d0/8.0d0*u(i)-1.0/8.0d0*u(i-2))**2
        td = td - (0.75d0*u(i)+3.0d0/8.0d0*u(i+1)-1.0/8.0d0*u(i-1))**2!Ref Wikipedia East face - West Face
		tc = 0.5d0*(2.0d0*td)/(x(i+1)-x(i-1))			!Multiply 2 due to cell face approx
		rhs(i) = 2*eps/(x(i+1)-x(i-1))*(ta-tb)+tc
	else
		ta = (u(i+1)-u(i))/(x(i+1)-x(i))
		tb = (u(i)-u(i-1))/(x(i)-x(i-1))
        td = (0.75d0*u(i)+3.0d0/8.0d0*u(i-1)-1.0/8.0d0*u(i+1))**2
        td = td - (0.75d0*u(i+1)+3.0d0/8.0d0*u(i)-1.0/8.0d0*u(i+2))**2
		tc = 0.5d0*(2.0d0*td)/(x(i+1)-x(i-1))			!Multiply 2 due to cell face approx
		rhs(i) = 2*eps/(x(i+1)-x(i-1))*(ta-tb)+tc
    end if
end do

do i = 2,nx-1
  nl(i) = -(u(i+1)-u(i-1))/(x(i+1)-x(i-1))*dxdt(i)
end do


do i = 2,nx-1
  v(i) = rhs(i)- nl(i)
end do

v(1) = 0.0d0
v(nx) = 0.0d0


end if


deallocate(rhs,nl)

return
end