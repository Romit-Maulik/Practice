!---------------------------------------------------------------------------!
!Burgers Equation Solver
!---------------------------------------------------------------------------!
!Periodic boundary condition in x in [0,1]
!
!		RK3 for time integration
!    	CS6 for spatial integration
!---------------------------------------------------------------------------!
!Case: Decaying Burger Turbulence
!      Approximate Deconvolution Modeling with Binomial Filters
!---------------------------------------------------------------------------!
!Omer San, Oklahoma State University, cfdlab.osu@gmail.com 
!Updated: Februay 16, 2016
!---------------------------------------------------------------------------!
program burger
implicit none
integer::nx,nt,ns,isp,np,im,NA,iord,idns,nr,inl,isf
real*8 ::nu,dt,tm,dx,t,pi,cfl,E0,neu,beta,cs,delta,afil,sfil
integer::i,j,k,freq,if1,if2,ifil
real*8,allocatable::u(:,:),uu(:),u1(:),u2(:),rr(:)

common /spectrum/ isp
common /les/ im,iord,idns
common /initial/ nr
common /nonlinear/ inl
common /filtering/ ifil
common /ADorder/ NA
common /viscosity/ nu
common /relaxation/ beta
common /mesh/ dx
common /smag/ cs,delta
common /pade/ afil

!reading input file
open(7,file='input.txt')
read(7,*)nx 	!number of grid points
read(7,*)nr 	!DNS/LES grid ration
read(7,*)ns 	!number of sample simulations
read(7,*)nu     !viscosity
read(7,*)dt     !time step
read(7,*)tm		!maximum time
read(7,*)isp    !choice of initial spectrum
read(7,*)np     !number of plots (equidistant in time)
read(7,*)im     ![1]DNS,[2]LES-AD,[3]LES-SMA,[4]LES-AD-SMA
read(7,*)NA     !order of ADM
read(7,*)beta   !relaxation paramater i.e., beta=1.0d0
read(7,*)delta   !smagorinsky stabilization radius
read(7,*)cs     !smagorinsky coefficient 
read(7,*)iord   ![6]6th-order CS,[4]4th-order CS
read(7,*)idns   ![1]filtered DNS, [2]unfiltered DNS
read(7,*)inl    ![1]convective,[2]conservative,[3]skew
read(7,*)ifil   !Binomial filters: [0]B2,[1]B4,[2]B6,[3]B8,[4]B21,[5]B31,[6]B41,[7]B22
read(7,*)afil   !Pade filter free parameter
read(7,*)isf    ![1]Apply seondary filter,[0]standard
read(7,*)sfil   !Secondary Pade filter free parameter to remove small scale noise
close(7)

if (im.eq.2) then
idns = 2  
end if


pi = 4.0d0*datan(1.0d0)

!spatial grid size
dx = 2.0d0*pi/dfloat(nx)

!smagorinsky stabilization radius
delta = delta*dx

!data array for velocity
allocate(u(0:nx,ns))
allocate(uu(0:nx))
allocate(u1(0:nx))
allocate(u2(0:nx))
allocate(rr(0:nx))


!compute initial conditions for all sample fields
call ic(nx,ns,dx,u)

!check for stability
111 continue
neu = nu*dt/(dx*dx)
cfl = maxval(u)*dt/dx
if (neu.ge.0.25d0) then
dt = 0.5d0*dt
goto 111
end if
if (cfl.ge.0.5d0) then
dt = 0.5d0*dt
goto 111
end if

!max number of time for numerical solution
nt = nint(tm/dt)

open(19,file='data.txt')
write(19,*) "dt =", dt
write(19,*) "Tm =", tm
write(19,*) "nu =", nu
write(19,*) "NT =", nt
close(19)

open(19,file='spectrum.plt')
write(19,*) 'variables ="k","E(k)"'

open(18,file='history.plt')
write(18,*) 'variables ="t","E(t)","D(t)","-dE/dt","P=-dE/dt-D"'

t = 0.0d0
if1=1000
if2=2000
call field(if1,nx,ns,dx,t,u) 
call aes(if2,nx,ns,t,u) 
if (isp.eq.0) then !from sine wave
E0 = 0.25d0 ! initial energy of sin(x) 
else
E0 = 0.5d0 !initial egery of given spectrums
end if

call history(nx,ns,nu,dt,E0,t,u) 


freq = int(nt/np)

!time integration for all sample fields
do j=1,nt

t = dfloat(j)*dt

  	if(isf.eq.1) then !secondary filtered
    
	do k=1,ns
    	
		do i=0,nx
        uu(i) = u(i,k)
        end do 	
        
		call rhs(nx,dx,nu,uu,rr)
    
		do i=0,nx
    	u1(i) = uu(i) + dt*rr(i)
    	end do
        
		call filterPade6P(nx,sfil,u1,u2)
		call rhs(nx,dx,nu,u2,rr)

		do i=0,nx
    	u2(i) = 0.75d0*uu(i) + 0.25d0*u2(i) + 0.25d0*dt*rr(i)
    	end do

		call filterPade6P(nx,sfil,u2,u1)
		call rhs(nx,dx,nu,u1,rr)

		do i=0,nx
    	uu(i) = 1.0d0/3.0d0*uu(i) + 2.0d0/3.0d0*u1(i) + 2.0d0/3.0d0*dt*rr(i)
    	end do

		call filterPade6P(nx,sfil,uu,u1)
		do i=0,nx
        u(i,k) = u1(i)
        end do        
              
	end do
    
  	else ! Standard
    
	do k=1,ns
    	
		do i=0,nx
        uu(i) = u(i,k)
        end do 	
        
		call rhs(nx,dx,nu,uu,rr)
    
		do i=0,nx
    	u1(i) = uu(i) + dt*rr(i)
    	end do
        
		call rhs(nx,dx,nu,u1,rr)

		do i=0,nx
    	u1(i) = 0.75d0*uu(i) + 0.25d0*u1(i) + 0.25d0*dt*rr(i)
    	end do

		call rhs(nx,dx,nu,u1,rr)

		do i=0,nx
    	uu(i) = 1.0d0/3.0d0*uu(i) + 2.0d0/3.0d0*u1(i) + 2.0d0/3.0d0*dt*rr(i)
    	end do
                  
		do i=0,nx
        u(i,k) = uu(i)
        end do
               
	end do

  	end if

    call history(nx,ns,nu,dt,E0,t,u) 


    !compute and plot velocity field
    !compute and plot energy spectrum
    if(mod(j,freq).eq.0) then
        if1=if1+1
        if2=if2+1
        call field(if1,nx,ns,dx,t,u) 
    	call aes(if2,nx,ns,t,u)    
    end if

	cfl = maxval(u)*dt/dx

    !check for numerical stability
	if (cfl.ge.1.0d0) then ! terminate computation and exit from the loop
        if1=if1+100
        if2=if2+100
        call field(if1,nx,ns,dx,t,u) 
    	call aes(if2,nx,ns,t,u)    
    
    goto 100
    end if
    
end do
100 continue

close(18)
close(19)

!plot final solution
open(8,file='final_fields.plt')
write(8,*) 'variables ="x","u(x)"'

do k=1,ns
	write(8,'(a16,i8,a10,i4,a3)')'zone f=point i=',nx+1,',t="case',k,'"'
	do i=0,nx
	write(8,*) dfloat(i)*dx,u(i,k)
	end do
end do

close(8)

end


!-----------------------------------------------------------------------------!
!compute rhs for numerical solutions
!  r = -u*u' + nu*u''
!-----------------------------------------------------------------------------!
subroutine rhs(nx,dx,nu,u,r)
implicit none
integer::nx,i,inl
real*8 ::nu,dx,cs,delta,u2(0:nx),up2(0:nx),s(0:nx),ss(0:nx)
real*8 ::u(0:nx),r(0:nx),up(0:nx),upp(0:nx),ua(0:nx),uap(0:nx),uu(0:nx),uuf(0:nx)
integer::im,iord,idns

common /les/ im,iord,idns
common /nonlinear/ inl
common /smag/ cs,delta


if (im.eq.1) then !DNS

  	if (inl.eq.1) then !convective
    
		if (iord.eq.4) then
			call c4dp(u,up,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
			call c6dp(u,up,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
			call c2dp(u,up,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if	
   
		do i=0,nx
		r(i) = -u(i)*up(i) + nu*upp(i)
		end do

  	else if(inl.eq.2) then !conservative
   
		do i=0,nx
		u2(i) = u(i)*u(i) 
		end do
     
		if (iord.eq.4) then
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if	
   
		do i=0,nx
		r(i) = -0.5d0*up2(i) + nu*upp(i)
		end do
  
  	else !skew
    
		do i=0,nx
		u2(i) = u(i)*u(i) 
		end do
     
		if (iord.eq.4) then
        	call c4dp(u,up,dx,nx)
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
        	call c6dp(u,up,dx,nx)
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
        	call c2dp(u,up,dx,nx)
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if	
   
		do i=0,nx
		r(i) = -(u(i)*up(i) + up2(i))/3.0d0 + nu*upp(i)
		end do

  	end if
      
      
else if (im.eq.2) then !LES-AD

    if (inl.eq.1) then !convective
  
		call adm(nx,u,ua)
   
		if (iord.eq.4) then
			call c4dp(ua,uap,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
			call c6dp(ua,uap,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
			call c2dp(ua,uap,dx,nx)
			call c2ddp(u,upp,dx,nx)     
		end if

		do i=0,nx
		uu(i) = ua(i)*uap(i)
		end do

		call filter(nx,uu,uuf)

		do i=0,nx
		r(i) = -uuf(i) + nu*upp(i)
		end do

    else if(inl.eq.2) then !conservative

    	call adm(nx,u,ua)

    	do i=0,nx
		u2(i) = ua(i)*ua(i) 
		end do
      
		if (iord.eq.4) then
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if

		call filter(nx,up2,uuf)

		do i=0,nx
		r(i) = -0.5d0*uuf(i) + nu*upp(i)
		end do  
    
	else !skew
    
    	call adm(nx,u,ua)

    	do i=0,nx
		u2(i) = ua(i)*ua(i) 
		end do
      
		if (iord.eq.4) then
        	call c4dp(ua,uap,dx,nx)
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
        	call c6dp(ua,uap,dx,nx)
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
        	call c2dp(ua,uap,dx,nx)
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if

		do i=0,nx
		uu(i) = (ua(i)*uap(i) + up2(i))/3.0d0
		end do
    
		call filter(nx,uu,uuf)

		do i=0,nx
		r(i) = -uuf(i) + nu*upp(i)
		end do 

    end if

else if (im.eq.3) then !LES-SMA

  	if (inl.eq.1) then !convective
    
		if (iord.eq.4) then
			call c4dp(u,up,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
			call c6dp(u,up,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
			call c2dp(u,up,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if	
   

		do i=0,nx
        s(i) = cs*cs*delta*delta*dabs(up(i))*up(i)
		end do
   
		if (iord.eq.4) then
			call c4dp(s,ss,dx,nx)		
		else if (iord.eq.6) then
			call c6dp(s,ss,dx,nx)		
    	else
			call c2dp(s,ss,dx,nx)		    
		end if
             		
		
		do i=0,nx
        !nue = cs*cs*delta*delta*dabs(up(i))
		!r(i) = -u(i)*up(i) + (nu+nue)*upp(i)
        r(i) = -u(i)*up(i) + nu*upp(i) + ss(i)
		end do

  	else if(inl.eq.2) then !conservative
   
		do i=0,nx
		u2(i) = u(i)*u(i) 
		end do
     
		if (iord.eq.4) then
        	call c4dp(u,up,dx,nx)
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
        	call c6dp(u,up,dx,nx)
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
        	call c2dp(u,up,dx,nx)
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if	
        
		do i=0,nx
        s(i) = cs*cs*delta*delta*dabs(up(i))*up(i)
		end do
   
		if (iord.eq.4) then
			call c4dp(s,ss,dx,nx)		
		else if (iord.eq.6) then
			call c6dp(s,ss,dx,nx)		
    	else
			call c2dp(s,ss,dx,nx)		    
		end if
           
		do i=0,nx
        !nue = cs*cs*delta*delta*dabs(up(i))
		!r(i) = -0.5d0*up2(i) + (nu+nue)*upp(i)
        r(i) = -0.5d0*up2(i) + nu*upp(i) + ss(i)
		end do
  
  	else !skew
    
		do i=0,nx
		u2(i) = u(i)*u(i) 
		end do
     
		if (iord.eq.4) then
        	call c4dp(u,up,dx,nx)
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
        	call c6dp(u,up,dx,nx)
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
        	call c2dp(u,up,dx,nx)
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if	
   
		do i=0,nx
        s(i) = cs*cs*delta*delta*dabs(up(i))*up(i)
		end do
   
		if (iord.eq.4) then
			call c4dp(s,ss,dx,nx)		
		else if (iord.eq.6) then
			call c6dp(s,ss,dx,nx)		
    	else
			call c2dp(s,ss,dx,nx)		    
		end if
        
		do i=0,nx
        !nue = cs*cs*delta*delta*dabs(up(i))
		!r(i) = -(u(i)*up(i) + up2(i))/3.0d0 + (nu+nue)*upp(i)
        r(i) = -(u(i)*up(i) + up2(i))/3.0d0 + nu*upp(i) + ss(i)
		end do

  	end if

else !LES-AD + SMA stabilization

    if (inl.eq.1) then !convective
  
		call adm(nx,u,ua)
   
		if (iord.eq.4) then
			call c4dp(ua,uap,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
			call c6dp(ua,uap,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
			call c2dp(ua,uap,dx,nx)
			call c2ddp(u,upp,dx,nx)     
		end if

		do i=0,nx
		uu(i) = ua(i)*uap(i)
		end do

		call filter(nx,uu,uuf)

		do i=0,nx
        s(i) = cs*cs*delta*delta*dabs(uap(i))*uap(i)
		end do
   
		if (iord.eq.4) then
			call c4dp(s,ss,dx,nx)		
		else if (iord.eq.6) then
			call c6dp(s,ss,dx,nx)		
    	else
			call c2dp(s,ss,dx,nx)		    
		end if
        
		do i=0,nx
        !nue = cs*cs*delta*delta*dabs(uap(i))
		!r(i) = -uuf(i) + (nu+nue)*upp(i)
        r(i) = -uuf(i) + nu*upp(i) + ss(i)
		end do

    else if(inl.eq.2) then !conservative

    	call adm(nx,u,ua)

    	do i=0,nx
		u2(i) = ua(i)*ua(i) 
		end do
      
		if (iord.eq.4) then
            call c4dp(ua,uap,dx,nx)
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
            call c6dp(ua,uap,dx,nx)
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
            call c2dp(ua,uap,dx,nx)
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if

		call filter(nx,up2,uuf)

		do i=0,nx
        s(i) = cs*cs*delta*delta*dabs(uap(i))*uap(i)        
		end do
   
		if (iord.eq.4) then
			call c4dp(s,ss,dx,nx)		
		else if (iord.eq.6) then
			call c6dp(s,ss,dx,nx)		
    	else
			call c2dp(s,ss,dx,nx)		    
		end if
        
		do i=0,nx
        !nue = cs*cs*delta*delta*dabs(uap(i))
		!r(i) = -0.5d0*uuf(i) + (nu+nue)*upp(i)
        r(i) = -0.5d0*uuf(i) + nu*upp(i) + ss(i)
		end do  
    
	else !skew
    
    	call adm(nx,u,ua)

    	do i=0,nx
		u2(i) = ua(i)*ua(i) 
		end do
      
		if (iord.eq.4) then
        	call c4dp(ua,uap,dx,nx)
			call c4dp(u2,up2,dx,nx)
			call c4ddp(u,upp,dx,nx)
		else if (iord.eq.6) then
        	call c6dp(ua,uap,dx,nx)
			call c6dp(u2,up2,dx,nx)
			call c6ddp(u,upp,dx,nx)
    	else
        	call c2dp(ua,uap,dx,nx)
			call c2dp(u2,up2,dx,nx)
			call c2ddp(u,upp,dx,nx)      
		end if

		do i=0,nx
		uu(i) = (ua(i)*uap(i) + up2(i))/3.0d0
		end do
    
		call filter(nx,uu,uuf)

		do i=0,nx
        s(i) = cs*cs*delta*delta*dabs(uap(i))*uap(i)
		end do
   
		if (iord.eq.4) then
			call c4dp(s,ss,dx,nx)		
		else if (iord.eq.6) then
			call c6dp(s,ss,dx,nx)		
    	else
			call c2dp(s,ss,dx,nx)		    
		end if
        
		do i=0,nx
        !nue = cs*cs*delta*delta*dabs(uap(i))
		!r(i) = -uuf(i) + (nu+nue)*upp(i)
        r(i) = -uuf(i) + nu*upp(i) + ss(i)
		end do 

    end if
    
end if

return
end


!------------------------------------------------------------------!
!Approximate deconvolution method for 1D data
!
!NA: order of van Cittert approximation
!
!unfiltered quantity u from from filtered variable uf
!by repeated filtering (van Cittert series)
!also known iterative inverse filtering
!
!filtering operation in physical space by Binomial filters
!------------------------------------------------------------------!
subroutine adm(n,uf,u)
implicit none
integer::n,NA,i,k
real*8::beta
real*8,dimension(0:n):: u,uf,ug

common /ADorder/ NA
common /relaxation/ beta


!initial guess
!k=0 
do i=0,n
u(i) =	beta*uf(i)
end do

!k>0    
do k = 1, NA    
    !compute filtered value of guess
    call filter(n,u,ug)  
	do i=0,n
		u(i) =	u(i) + beta*(uf(i) - ug(i))
	end do	
end do



return
end



!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data
!Filtering performed in physical space by Binomial filters
!Periodic data
!---------------------------------------------------------------------------!
subroutine filter(nx,u,uf)
implicit none
integer::nx,ifil
real*8 ::u(0:nx),uf(0:nx)
real*8 ::afil

common /filtering/ ifil
common /pade/ afil

!Binomial filters: [0]B2,[1]B4,[2]B6,[3]B8,[4]B21,[5]B31,[6]B41,[7]B22

if (ifil.eq.0) then
	!Binomial Filter: B2
	call filterB2(nx,u,uf)
    
else if (ifil.eq.1) then
	!Binomial Filter: B4
	call filterB4(nx,u,uf)

else if (ifil.eq.2) then
	!Binomial Filter: B6
	call filterB6(nx,u,uf)

else if (ifil.eq.3) then
	!Binomial Filter: B8
	call filterB8(nx,u,uf) 

else if (ifil.eq.4) then
	!Binomial Filter: B21  (n=2, l=1)
	call filterB21(nx,u,uf)

else if (ifil.eq.5) then
	!Binomial Filter: B31  (n=3, l=1)
	call filterB31(nx,u,uf)

else if (ifil.eq.6) then
	!Binomial Filter: B41  (n=4, l=1)
	call filterB41(nx,u,uf)

else if (ifil.eq.7) then
	!Binomial Filter: B22  (n=2, l=2)
	call filterB22(nx,u,uf) 

else
    !Pade filter
    call filterPade6P(nx,afil,u,uf)
end if


return
end



!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B2
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB2(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-1:nx+1)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)

cc = 1.0d0/4.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(v(i-1) + 2.0d0*v(i) + v(i+1))
end do

return 
end


!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B4
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB4(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-2:nx+2)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(-2) = v(nx-2) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)
v(nx+2) = v(2)  !periodic bc (ghost point)

cc = 1.0d0/16.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(v(i-2) + 4.0d0*v(i-1) + 6.0d0*v(i) + 4.0d0*v(i+1) + v(i+2))
end do

return 
end

!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B6
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB6(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-3:nx+3)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(-2) = v(nx-2) !periodic bc (ghost point)
v(-3) = v(nx-3) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)
v(nx+2) = v(2)  !periodic bc (ghost point)
v(nx+3) = v(3)  !periodic bc (ghost point)

cc = 1.0d0/64.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(v(i-3) + 6.0d0*v(i-2) + 15.0d0*v(i-1) + 20.0d0*v(i) &
        + 15.0d0*v(i+1) + 6.0d0*v(i+2) + v(i+3))
end do

return 
end

!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B8
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB8(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-4:nx+4)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(-2) = v(nx-2) !periodic bc (ghost point)
v(-3) = v(nx-3) !periodic bc (ghost point)
v(-4) = v(nx-4) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)
v(nx+2) = v(2)  !periodic bc (ghost point)
v(nx+3) = v(3)  !periodic bc (ghost point)
v(nx+4) = v(4)  !periodic bc (ghost point)

cc = 1.0d0/256.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(v(i-4) + 8.0d0*v(i-3) + 28.0d0*v(i-2) + 56.0d0*v(i-1) + 70.0d0*v(i) &
        + 56.0d0*v(i+1) + 28.0d0*v(i+2) + 8.0d0*v(i+3) + v(i+4))
end do

return 
end

!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B21
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB21(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-2:nx+2)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(-2) = v(nx-2) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)
v(nx+2) = v(2)  !periodic bc (ghost point)

cc = 1.0d0/16.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(-v(i-2) + 4.0d0*v(i-1) + 10.0d0*v(i) + 4.0d0*v(i+1) - v(i+2))
end do

return 
end

!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B31
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB31(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-3:nx+3)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(-2) = v(nx-2) !periodic bc (ghost point)
v(-3) = v(nx-3) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)
v(nx+2) = v(2)  !periodic bc (ghost point)
v(nx+3) = v(3)  !periodic bc (ghost point)

cc = 1.0d0/64.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(v(i-3) - 6.0d0*v(i-2) + 15.0d0*v(i-1) + 44.0d0*v(i) &
        + 15.0d0*v(i+1) - 6.0d0*v(i+2) + v(i+3))
end do

return 
end

!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B41
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB41(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-4:nx+4)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(-2) = v(nx-2) !periodic bc (ghost point)
v(-3) = v(nx-3) !periodic bc (ghost point)
v(-4) = v(nx-4) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)
v(nx+2) = v(2)  !periodic bc (ghost point)
v(nx+3) = v(3)  !periodic bc (ghost point)
v(nx+4) = v(4)  !periodic bc (ghost point)

cc = 1.0d0/256.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(-v(i-4) + 8.0d0*v(i-3) - 28.0d0*v(i-2) + 56.0d0*v(i-1) + 186.0d0*v(i) &
        + 56.0d0*v(i+1) - 28.0d0*v(i+2) + 8.0d0*v(i+3) - v(i+4))
end do

return 
end


!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data using a discrete filter
!Binomial Filter: B22
!Periodic data
!---------------------------------------------------------------------------!
subroutine filterB22(nx,u,uf)
implicit none
integer::nx,i
real*8 ::u(0:nx),uf(0:nx), v(-4:nx+4)
real*8 ::cc

! u: given function
! uf: filtered function
! v: same as u with ghost point (using periodic bc)

do i=0,nx
v(i) = u(i)     !internal data
end do
v(-1) = v(nx-1) !periodic bc (ghost point)
v(-2) = v(nx-2) !periodic bc (ghost point)
v(-3) = v(nx-3) !periodic bc (ghost point)
v(-4) = v(nx-4) !periodic bc (ghost point)
v(nx+1) = v(1)  !periodic bc (ghost point)
v(nx+2) = v(2)  !periodic bc (ghost point)
v(nx+3) = v(3)  !periodic bc (ghost point)
v(nx+4) = v(4)  !periodic bc (ghost point)

cc = 1.0d0/256.0d0
!Compute filtered quantities
do i=0,nx
uf(i)=cc*(v(i-4) - 8.0d0*v(i-3) - 4.0d0*v(i-2) + 72.0d0*v(i-1) + 134.0d0*v(i) &
        + 72.0d0*v(i+1) - 4.0d0*v(i+2) - 8.0d0*v(i+3) + v(i+4))
end do

return 
end


!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data
!
!Filtering performed in Fourier space
!
!Periodic data
!
!
!Pade sixth-order:  -0.5 < afil < 0.5
!---------------------------------------------------------------------------!
subroutine filterPade6F(nx,afil,u,uf)
implicit none
integer::nx
real*8 ::afil
real*8 ::u(0:nx),uf(0:nx)
real*8 ::kx(0:nx-1) !wave number
real*8 ::data1d(2*nx) !both real and imaginary parts
integer::i,k,isign,ndim,nn(1)
real*8 ::pi,kr,tf,wx,d1,d2

!One-dimensional data array
ndim =1
nn(1)=nx

pi = 4.0d0*datan(1.0d0)
wx = 2.0d0*pi/dfloat(nx)

!find the data coefficient in Fourier space
k=1
do i=0,nx-1   
	data1d(k)   =  u(i)
	data1d(k+1) =  0.0d0    
k = k + 2
end do
!normalize
do k=1,2*nx
data1d(k)=data1d(k)/dfloat(nx)
end do
!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)

!Wave numbers: 0,1,2,...,nx/2-1, -nx/2, -nx/2+1, ...,-1 
do i=0,nx/2-1
kx(i)      = dfloat(i)
kx(i+nx/2) = dfloat(i-nx/2)
end do

!Filter according to transfer function
!Apply transfer function in Fourier space 
 
k=1
do i=0,nx-1 
kr = dabs(kx(i))
d2 = (1.0d0+2.0d0*afil*dcos(kr*wx))
d1 = (11.0d0/16.0d0 + 5.0d0/8.0d0*afil) &
    +(15.0d0/32.0d0 + 17.0d0/16.0d0*afil)*dcos(kr*wx)  &
    +(-3.0d0/16.0d0 + 3.0d0/8.0d0*afil)*dcos(2.0d0*kr*wx) &
    +(1.0d0/32.0d0 - 1.0d0/16.0d0*afil)*dcos(3.0d0*kr*wx) 
tf = d1/d2
data1d(k)   = data1d(k)*tf
data1d(k+1) = data1d(k+1)*tf
k = k + 2
end do


!find the data in physical space
!forward fourier transform
isign= 1
call fourn(data1d,nn,ndim,isign)

!Compute back the filtered quantities
k=1
do i=0,nx-1
uf(i)=data1d(k)
k=k+2
end do

! periodicity
uf(nx)=uf(0)

return 
end



!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data
!
!Filtering performed in physical space
!
!Periodic data
!
!
!Pade sixth-order:  -0.5 < afil < 0.5
!
!---------------------------------------------------------------------------!
subroutine filterPade6P(nx,afil,u,uf)
implicit none
integer::nx,i
real*8 ::afil
real*8 ::u(0:nx),uf(0:nx)
real*8 ::alpha,beta
real*8, dimension (0:nx-1):: a,b,c,r,x 

do i=0,nx-1
a(i) = afil
b(i) = 1.0d0
c(i) = afil
end do

do i=3,nx-3
r(i) = (11.0d0/16.0d0 + 5.0d0/8.0d0*afil)*u(i) &
      +(15.0d0/32.0d0 + 17.0d0/16.0d0*afil)*0.5d0*(u(i-1)+u(i+1))  &
      +(-3.0d0/16.0d0 + 3.0d0/8.0d0*afil)*0.5d0*(u(i-2)+u(i+2)) &
      +(1.0d0/32.0d0 - 1.0d0/16.0d0*afil)*0.5d0*(u(i-3)+u(i+3))
end do

r(2) = (11.0d0/16.0d0 + 5.0d0/8.0d0*afil)*u(2) &
      +(15.0d0/32.0d0 + 17.0d0/16.0d0*afil)*0.5d0*(u(2-1)+u(2+1))  &
      +(-3.0d0/16.0d0 + 3.0d0/8.0d0*afil)*0.5d0*(u(2-2)+u(2+2)) &
      +(1.0d0/32.0d0 - 1.0d0/16.0d0*afil)*0.5d0*(u(nx+2-3)+u(2+3))
      
r(1) = (11.0d0/16.0d0 + 5.0d0/8.0d0*afil)*u(1) &
      +(15.0d0/32.0d0 + 17.0d0/16.0d0*afil)*0.5d0*(u(1-1)+u(1+1))  &
      +(-3.0d0/16.0d0 + 3.0d0/8.0d0*afil)*0.5d0*(u(nx+1-2)+u(1+2)) &
      +(1.0d0/32.0d0 - 1.0d0/16.0d0*afil)*0.5d0*(u(nx+1-3)+u(1+3))

r(nx-1) = (11.0d0/16.0d0 + 5.0d0/8.0d0*afil)*u(nx-1) &
      +(15.0d0/32.0d0 + 17.0d0/16.0d0*afil)*0.5d0*(u(nx-1-1)+u(nx-1+1))  &
      +(-3.0d0/16.0d0 + 3.0d0/8.0d0*afil)*0.5d0*(u(nx-1-2)+u(-1+2)) &
      +(1.0d0/32.0d0 - 1.0d0/16.0d0*afil)*0.5d0*(u(nx-1-3)+u(-1+3))

r(nx-2) = (11.0d0/16.0d0 + 5.0d0/8.0d0*afil)*u(nx-2) &
      +(15.0d0/32.0d0 + 17.0d0/16.0d0*afil)*0.5d0*(u(nx-2-1)+u(nx-2+1))  &
      +(-3.0d0/16.0d0 + 3.0d0/8.0d0*afil)*0.5d0*(u(nx-2-2)+u(-2+2)) &
      +(1.0d0/32.0d0 - 1.0d0/16.0d0*afil)*0.5d0*(u(nx-2-3)+u(-2+3))
      
r(0) = (11.0d0/16.0d0 + 5.0d0/8.0d0*afil)*u(0) &
      +(15.0d0/32.0d0 + 17.0d0/16.0d0*afil)*0.5d0*(u(nx+0-1)+u(0+1))  &
      +(-3.0d0/16.0d0 + 3.0d0/8.0d0*afil)*0.5d0*(u(nx+0-2)+u(0+2)) &
      +(1.0d0/32.0d0 - 1.0d0/16.0d0*afil)*0.5d0*(u(nx+0-3)+u(0+3))
      
     
alpha = afil
beta  = afil

call ctdms(a,b,c,alpha,beta,r,x,0,nx-1) 

do i=0,nx-1
uf(i)=x(i)
end do
uf(nx)=uf(0)

return 
end


!---------------------------------------------------------------------------!
!History
!---------------------------------------------------------------------------!
subroutine history(nx,ns,nu,dt,E0,t,u)
implicit none
integer::nx,ns,i,k
real*8 ::t,nu,dt,u(0:nx,ns),uu(0:nx),uuf(0:nx),ue(1:nx/2-1),ua(1:nx/2-1)
real*8 ::EE,DD,D2,E0
integer::im,iord,idns

common /les/ im,iord,idns



	do i=1,nx/2-1
	ua(i)=0.0d0
	end do
        
do k=1,ns
          
    do i=0,nx
    uu(i)=u(i,k)
    end do
    
	if (idns.eq.1) then
		call filter(nx,uu,uuf)
    	call field2spec1d(nx,uuf,ue)
    else
    	call field2spec1d(nx,uu,ue)  
	end if
    
    
        
	do i=1,nx/2-1
    ua(i)= ua(i) + ue(i)
    end do
        
end do

    do i=1,nx/2-1
    ua(i)=ua(i) / dfloat(ns)
    end do

!multiply by 2 (-+data)
EE = 0.0d0
DD = 0.0d0
do i=1,nx/2-1
EE = EE + 2.0d0*ua(i)
DD = DD + 4.0d0*nu*(dfloat(i)*dfloat(i))*ua(i)
end do

D2 = (E0 - EE)/dt 
E0 = EE

if(t.le.0.1d0*dt)then !only for initial step before dE/dt available
D2 = DD
end if

!Plot
write(18,*) t,EE,DD,D2,D2-DD


return
end


!---------------------------------------------------------------------------!
!Write field
!---------------------------------------------------------------------------!
subroutine field(if1,nx,ns,dx,t,u)
implicit none
integer::nx,ns,i,j,k,if1
real*8 ::dx,nu,t,u(0:nx,ns),uu(0:nx),uuf(0:nx)
real*8 ::sf2(0:nx/2),g(0:nx/2),up(0:nx)
integer::im,iord,idns,ir

common /les/ im,iord,idns
common /viscosity/ nu

!only for first data array
k=1
	do i=0,nx                  
    uu(i)= u(i,k)
    end do  

    if (idns.eq.1) then
		call filter(nx,uu,uuf) 
        do i=0,nx                  
    	uu(i)= uuf(i)
    	end do  	   	 
	end if


      
!Plot field
open(if1)
write(if1,*) 'variables ="x","u(x)"'
write(if1,'(a16,i8,a10,f10.4,a3)')'zone f=point i=',nx+1,',t="time',t,'"'
do i=0,nx
write(if1,*) dfloat(i)*dx,uu(i)
end do
close(if1)


!Compute 2nd-order structure function

do j=0,nx/2

	sf2(j) = 0.0d0

    !ensamble over fields
    do k=1,ns

		do i=0,nx-1
        
            ir = i + j
			if(ir.ge.nx) then
            ir=ir-nx
            end if
  
        	sf2(j) = sf2(j) + (u(ir,k)-u(i,k))*(u(ir,k)-u(i,k))
        end do
    
    end do

    !normalizing

    sf2(j) = sf2(j) / dfloat(ns*nx)
    
end do


!Plot field
open(if1+5000)
write(if1+5000,*) 'variables ="r","SF2(r)"'
write(if1+5000,'(a16,i8,a10,f10.4,a3)')'zone f=point i=',nx/2+1,',t="time',t,'"'
do i=0,nx/2
write(if1+5000,*) dfloat(i)*dx,sf2(i)
end do
close(if1+5000)



!Compute dissipation rate correlation


do j=0,nx/2

	g(j) = 0.0d0

    !ensamble over fields
    do k=1,ns

	do i=0,nx                  
    uu(i)= u(i,k)
    end do  

    if (iord.eq.4) then
		call c4dp(uu,up,dx,nx)
	else if (iord.eq.6) then
		call c6dp(uu,up,dx,nx)
    else
		call c2dp(uu,up,dx,nx)  
	end if
    
		do i=0,nx-1
        
            ir = i + j
			if(ir.ge.nx) then
            ir=ir-nx
            end if
  
        	g(j) = g(j) + (nu*up(ir)*up(i))*(nu*up(ir)*up(i))
        end do
    
    end do

    !normalizing

    g(j) = g(j) / dfloat(ns*nx)
    
end do

!Plot field
open(if1+6000)
write(if1+6000,*) 'variables ="r","g(r)"'
write(if1+6000,'(a16,i8,a10,f10.4,a3)')'zone f=point i=',nx/2+1,',t="time',t,'"'
do i=0,nx/2
write(if1+6000,*) dfloat(i)*dx,g(i)
end do
close(if1+6000) 


return
end

!---------------------------------------------------------------------------!
!Write averaged energy specrum
!---------------------------------------------------------------------------!
subroutine aes(if2,nx,ns,t,u)
implicit none
integer::nx,ns,i,k,if2
real*8 ::t,u(0:nx,ns),uu(0:nx),uuf(0:nx),ue(1:nx/2-1),ua(1:nx/2-1)
integer::im,iord,idns

common /les/ im,iord,idns
 

	do i=1,nx/2-1
	ua(i)=0.0d0
	end do
        
do k=1,ns
          
    do i=0,nx
    uu(i)=u(i,k)
    end do

	if (idns.eq.1) then
		call filter(nx,uu,uuf)
    	call field2spec1d(nx,uuf,ue)
    else
    	call field2spec1d(nx,uu,ue)
	end if
               
        
	do i=1,nx/2-1
    ua(i)= ua(i) + ue(i)
    end do
        
end do

    do i=1,nx/2-1
    ua(i)=ua(i) / dfloat(ns)
    end do
     
!Plot energy spectrum
open(if2)
write(if2,*) 'variables ="k","E(k)"'
write(if2,'(a16,i8,a10,f10.4,a3)')'zone f=point i=',nx/2-1,',t="time',t,'"'
do i=1,nx/2-1
write(if2,*) dfloat(i),ua(i)
end do
close(if2)

!Plot energy spectrum the spectrum.plt
write(19,'(a16,i8,a10,f10.4,a3)')'zone f=point i=',nx/2-1,',t="time',t,'"'
do i=1,nx/2-1
write(19,*) dfloat(i),ua(i)
end do


return
end



!---------------------------------------------------------------------------!
!Initial conditions
!---------------------------------------------------------------------------!
subroutine ic(nx,ns,dx,u)
implicit none
integer::nx,ns,i,k,isp,nr,im,iord,idns
real*8 ::dx,dx1,u(0:nx,ns),v(0:nx),vf(0:nx)
real*8,allocatable ::uu(:)

common /spectrum/ isp
common /initial/ nr
common /les/ im,iord,idns

if (isp.eq.0) then !deterministic (sin wave)
do k=1,ns   
    do i=0,nx
    u(i,k) = dsin(1.0d0*dfloat(i)*dx)
    end do    
end do

else !from spectrum

allocate(uu(0:nx*nr))
dx1 = dx/dfloat(nr)
  
do k=1,ns
	call spec2field1d(nx*nr,dx1,uu)    
    do i=0,nx
    u(i,k) = uu(i*nr)
    end do    
end do

end if


!filtered initial conditions if LES
if (im.eq.2) then

	do k=1,ns
          
    	do i=0,nx
    	v(i)=u(i,k)
    	end do
    		
		call filter(nx,v,vf)  
        
		do i=0,nx
    	u(i,k)= vf(i)
    	end do
        
	end do

end if
 
open(8,file='initial_fields.plt')
write(8,*) 'variables ="x","u(x)"'

do k=1,ns
	write(8,'(a16,i8,a10,i4,a3)')'zone f=point i=',nx+1,',t="case',k,'"'
	do i=0,nx
	write(8,*) dfloat(i)*dx,u(i,k)
	end do
end do

close(8)


return
end



!---------------------------------------------------------------------------!
!Compute 1D energy spectra from the velocity field
!Periodic, equidistant grid
!---------------------------------------------------------------------------!
subroutine field2spec1d(nx,u,ue)
implicit none
integer::nx
real*8 ::u(0:nx),ue(1:nx/2-1)
real*8 ::es(0:nx-1) !energy density
real*8 ::data1d(2*nx) !both real and imaginary parts
integer::i,k,isign,ndim,nn(1)

ndim =1
nn(1)=nx

!find the velocity in Fourier space
k=1
do i=0,nx-1   
	data1d(k)   =  u(i)
	data1d(k+1) =  0.0d0    
k = k + 2
end do
!normalize
do k=1,2*nx
data1d(k)=data1d(k)/dfloat(nx)
end do
!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)
     
!Energy spectrum
k=1
do i=0,nx-1   
es(i) = 0.5d0*(data1d(k)*data1d(k) + data1d(k+1)*data1d(k+1))
k = k + 2
end do

!Angle avaraged energy spectrum
do i=1,nx/2-1
ue(i) = 0.5d0*(es(i)+es(nx-i))
end do


return
end







!---------------------------------------------------------------------------!
!Compute 1D velocity field from the given energy spectrum as E1(k)
!Periodic, equidistant grid
!---------------------------------------------------------------------------!
subroutine spec2field1d(nx,dx,u)
implicit none
integer::nx
real*8 ::dx,u(0:nx)
real*8 ::ran,Lx,pi,kk,E1
real*8 ::kx(0:nx-1)   !wave number
real*8 ::data1d(2*nx) !both real and imaginary parts
real*8 ::phase(2*nx)  !both real and imaginary parts
integer::i,k,isign,ndim,nn(1),iseed

ndim =1
nn(1)=nx

!Set seed for the random number generator between [0,1]
iseed = 19
CALL RANDOM_SEED(iseed)

pi = 4.0d0*datan(1.0d0)
Lx = dfloat(nx)*dx

!Wave numbers 
do i=0,nx/2-1
kx(i)      = 2.0d0*pi/lx*dfloat(i)
kx(i+nx/2) = 2.0d0*pi/lx*dfloat(i-nx/2)
end do

!Random phase
CALL RANDOM_NUMBER(ran) 
phase(1)      =  dcos(2.0d0*pi*ran) 
phase(2)      =  0.0d0
phase(nx+1)   =  dcos(2.0d0*pi*ran) 
phase(nx+2)   =  0.0d0
k=3
do i=1,nx/2-1
	CALL RANDOM_NUMBER(ran) 
	phase(k)        =  dcos(2.0d0*pi*ran) 
	phase(k+1)      =  dsin(2.0d0*pi*ran) 
    phase(2*nx-k+2) =  dcos(2.0d0*pi*ran) 
    phase(2*nx-k+3) = -dsin(2.0d0*pi*ran) 
k = k + 2
end do

        
!Vecocity amplitudes in Fourier space 
k=1
do i=0,nx-1   
    kk = dabs(kx(i))
	data1d(k)   =  dsqrt(2.0d0*E1(kk))*phase(k)
	data1d(k+1) =  dsqrt(2.0d0*E1(kk))*phase(k+1)    
k = k + 2
end do

!find the velocity in physical space
!forward fourier transform
isign= 1
call fourn(data1d,nn,ndim,isign)

k=1
do i=0,nx-1
u(i)=data1d(k)
k=k+2
end do

! periodicity
u(nx)=u(0)

return
end


!---------------------------------------------------------------------------!
!Given energy spectrum
!
!Designed to have Etot = integral of E(k) = 1/2
!---------------------------------------------------------------------------!
real*8 function E1(kr)
implicit none
integer::isp
real*8 :: kr,a,b,k0,pi

common /spectrum/ isp

if (isp.eq.1) then
  
k0 = 2.0d0
pi = 4.0d0*datan(1.0d0)
b  = 0.5d0*k0*k0
a  = 4.0d0*dsqrt(b)/(3.0d0*pi)
E1 = (a*kr**4.0d0)/((b+kr*kr)**3.0d0)

else !used the following in the paper (IJCFD 2016)

k0 = 10
pi = 4.0d0*datan(1.0d0)
a  = 2.0d0/(3.0d0*dsqrt(pi))/(k0**5.0d0)
E1 = a*(kr**4.0d0)*dexp(-(kr/k0)**2.0d0)

end if


end



!---------------------------------------------------------------------------!
!N-dimensional FFT routine 
!only for power of 2
!by Numerical Recipes
!i.e., for two dimensional problems ndim = 2
!nn(1)=nx
!nn(2)=ny
!---------------------------------------------------------------------------!
subroutine fourn(data,nn,ndim,isign)
implicit none
integer::ndim,isign
integer::nn(ndim)
real*8:: wr,wi,wpr,wpi,wtemp,theta
integer::i1,i2,i2rev,i3,i3rev,ibit,idim,ifp1,ifp2,ip1,ip2,ip3
integer::k1,k2,n,nprev,nrem,ntot
real*8:: tempr,tempi
real*8:: data(*)

ntot=1
do idim=1,ndim
ntot=ntot*nn(idim)
end do
nprev=1
do idim=1,ndim
        n=nn(idim)
        nrem=ntot/(n*nprev)
        ip1=2*nprev
        ip2=ip1*n
        ip3=ip2*nrem
        i2rev=1
        do i2=1,ip2,ip1
          if(i2.lt.i2rev)then
            do i1=i2,i2+ip1-2,2
              do i3=i1,ip3,ip2
                i3rev=i2rev+i3-i2
                tempr=data(i3)
                tempi=data(i3+1)
                data(i3)=data(i3rev)
                data(i3+1)=data(i3rev+1)
                data(i3rev)=tempr
                data(i3rev+1)=tempi
			  end do
			end do
          endif
          ibit=ip2/2
		  1 continue
          if ((ibit.ge.ip1).and.(i2rev.gt.ibit)) then
            i2rev=i2rev-ibit
            ibit=ibit/2
          go to 1
          endif
          i2rev=i2rev+ibit
		end do
        ifp1=ip1
		2 continue
        if(ifp1.lt.ip2)then
          ifp2=2*ifp1
          theta=isign*6.28318530717959d0/(ifp2/ip1)
          wpr=-2.d0*dsin(0.5d0*theta)**2
          wpi=dsin(theta)
          wr=1.d0
          wi=0.d0
          do i3=1,ifp1,ip1
            do i1=i3,i3+ip1-2,2
              do i2=i1,ip3,ifp2
                k1=i2
                k2=k1+ifp1
                tempr=sngl(wr)*data(k2)-sngl(wi)*data(k2+1)
                tempi=sngl(wr)*data(k2+1)+sngl(wi)*data(k2)
                data(k2)=data(k1)-tempr
                data(k2+1)=data(k1+1)-tempi
                data(k1)=data(k1)+tempr
                data(k1+1)=data(k1+1)+tempi
			  end do
			end do
            wtemp=wr
            wr=wr*wpr-wi*wpi+wr
            wi=wi*wpr+wtemp*wpi+wi
		  end do
          ifp1=ifp2
        go to 2
        endif
nprev=n*nprev
end do
return
end


!------------------------------------------------------------------!
! c2dp:  2nd-order central scheme for first-degree derivative(up)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c2dp(u,up,h,n)
implicit none
integer :: n,i
real*8  :: h
real*8, dimension (0:n)  :: u,up

do i=1,n-1
up(i) =(u(i+1)-u(i-1))/(2.0d0*h)
end do
i=0
up(i) =(u(i+1)-u(i-1+n))/(2.0d0*h)
i=n
up(i) =(u(i+1-n)-u(i-1))/(2.0d0*h)

return
end

!------------------------------------------------------------------!
! c2ddp: 2nd-order central scheme for second-degree derivative(upp)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c2ddp(u,upp,h,n)
implicit none
integer :: n,i
real*8  :: h
real*8, dimension (0:n)  :: u,upp

do i=1,n-1
upp(i) =(u(i+1) - 2.0d0*u(i) + u(i-1))/(h*h)
end do
i=0
upp(i) =(u(i+1) - 2.0d0*u(i) + u(i-1+n))/(h*h)
i=n
upp(i) =(u(i+1-n) - 2.0d0*u(i) + u(i-1))/(h*h)

return
end

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
real*8  :: h,alpha,beta
real*8, dimension (0:n)  :: u,up
real*8, dimension (0:n-1):: a,b,c,r,x 

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
real*8  :: h,alpha,beta
real*8, dimension (0:n)  :: u,up
real*8, dimension (0:n-1):: a,b,c,r,x 

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
real*8  :: h,alpha,beta
real*8, dimension (0:n)  :: u,upp
real*8, dimension (0:n-1):: a,b,c,r,x 

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
real*8  :: h,alpha,beta
real*8, dimension (0:n)  :: u,upp
real*8, dimension (0:n-1):: a,b,c,r,x 


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
real*8 ::a(s:e),b(s:e),c(s:e),r(s:e),u(s:e) 
integer::j  
real*8 ::bet,gam(s:e) 
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
real*8 :: alpha,beta,a(s:e),b(s:e),c(s:e),r(s:e),x(s:e)  
integer:: i  
real*8 :: fact,gamma,bb(s:e),u(s:e),z(s:e)
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
!Simpson's 1/3 rule for numerical integration of g(i)
!for equally distributed mesh with interval dx 
!nx should be even number
!----------------------------------------------------------!
subroutine simp1D(nx,dx,g,s)
implicit none
integer::nx,i,nh
real*8 ::dx,g(0:nx),s,ds,th

	nh = int(nx/2)
	th = 1.0d0/3.0d0*dx
    
	s  = 0.0d0
	do i=0,nh-1
	ds = th*(g(2*i)+4.0d0*g(2*i+1)+g(2*i+2))
	s  = s + ds
	end do

return
end