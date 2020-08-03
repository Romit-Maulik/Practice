!------------------------------------------------------------------------------!
!>>> Euler/NS Solver in 3D Cartesian Domain >>>
!------------------------------------------------------------------------------!
!Parallel Code Implementation with MPI
!To run in Cowboy Cluster (OSU): 
!          	module load openmpi-1.4/intel
!          	mpif90 euler3d_MPI.f90
!		   	qsub mexpress.job (mbatch.job or mlong.job)
!------------------------------------------------------------------------------!
!Schemes: 	5th order WENO reconstruction (more options are available)
!         	Rusanov Riemann solver at interface (more options are available)
!         	3rd order TVD RK for time integration
!------------------------------------------------------------------------------!
!BCs: 		Periodic boundary conditions 
!------------------------------------------------------------------------------!
!Domain: 	In a cubix-box [x:0,2pi]*[y:0,2pi]*[z:0,2pi]
!			In a unit-box [x:-0.5,0.5]*[y:-0.5,0.5]*[z:-0.5,0.5]			
!------------------------------------------------------------------------------!
!Cases: [1] Taylor Green Vortex (TGV) Problem
!		[2] Kida-Pelz Vortex (KPV) Problem
!		[3] Kelvin-Helmholtz Instability (KHI) Problem
!------------------------------------------------------------------------------!
!References:San, Kara, Computers and Fluids, Vol 117, 2015
!			Boratav-Pelz, Physics of Fluids, Vol 6, 1994
!			Hickel, Adams, Domaradzki, JCP, Vol 213, 2006 
!			Bull, Jameson, AIAA Journal, Vol 53, 2015 
!------------------------------------------------------------------------------!
!CFDLab.org, cfdlab.osu@gmail.com
!Oklahoma State University, Stillwater
!Updated: May 15, 2017
!------------------------------------------------------------------------------!
program euler3D
implicit none
include 'mpif.h'	!MPI header file

!Problem related variables
integer:: nx_global,ny_global,nz_global,nsend
integer:: nx,ny,nz,nt,nsnap,ifile,idt,nf,isgs,ifil,imodel,iwriteI,iwriteF,iwriteS
real*8 :: cfl,dt,dx,dy,dz,lx,ly,lz,time,tmax,eps,epz,t1,t2,te,rate,te0
real*8 :: gamma,a,b,z_min,sigma,re,pr,pi,Tref,ma,te_local,dt_local,x0,y0,z0
real*8 :: ukhi,denr,tout,dtout,alpha,ampy
integer:: i,j,k,m,n,ic,ix,nz1,jip,iweno,iupw,pweno,ipr,iout,iend
real*8,allocatable:: q(:,:,:,:),u(:,:,:,:),s(:,:,:,:),x(:,:,:),y(:,:,:),z(:,:,:)

!MPI related variables
integer               :: myid
integer               :: np
integer               :: ierr
integer, dimension(60):: req = MPI_REQUEST_NULL
integer               :: status(MPI_STATUS_SIZE)
integer               :: status_array(MPI_STATUS_SIZE,60) !total 60 MPI requests of send/receive (12*5)
integer, parameter    :: id_top2bottom_1 = 1000 !message tag 
integer, parameter    :: id_bottom2top_1 = 1001 !message tag
integer, parameter    :: id_top2bottom_2 = 2000 !message tag 
integer, parameter    :: id_bottom2top_2 = 2001 !message tag
integer, parameter    :: id_top2bottom_3 = 3000 !message tag 
integer, parameter    :: id_bottom2top_3 = 3001 !message tag

common /fluids/ gamma
common /mach/ ma
common /weno_power/ pweno
common /weno_epsilon_j/ eps
common /weno_epsilon_z/ epz
common /modeling/ imodel,isgs
common /viscosity/ re,pr,Tref
common /filterstrength/ sigma
common /fluxopt/ ix
common /weno_opt/ iweno
common /upw_opt/ iupw
common /KHIconstants/ ukhi,denr,alpha,ampy

open(10,file='input.txt')
read(10,*)nx_global
read(10,*)ipr
read(10,*)ukhi
read(10,*)denr
read(10,*)cfl
read(10,*)tmax
read(10,*)nsnap
read(10,*)eps
read(10,*)epz
read(10,*)pweno
read(10,*)gamma
read(10,*)dt
read(10,*)idt
read(10,*)imodel
read(10,*)isgs
read(10,*)iweno
read(10,*)iupw
read(10,*)ifil
read(10,*)sigma
read(10,*)re
read(10,*)pr
read(10,*)ma
read(10,*)Tref
read(10,*)iwriteI
read(10,*)iwriteF
read(10,*)iwriteS
read(10,*)ix
read(10,*)alpha
read(10,*)ampy
close(10)



177 continue ! to check with domain restriction
!----------------------------------------------------------------------!
! MPI initialize
!----------------------------------------------------------------------!
      call MPI_INIT(ierr) 

      call MPI_Comm_size(MPI_COMM_WORLD, np, ierr) 

      call MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)
      
!----------------------------------------------------------------------!
!Global domain:
!----------------------------------------------------------------------!
pi = 4.0d0*datan(1.0d0)

if (ipr.eq.3) then !KHI flow domain 
!a cartesian box domain for KHI flow 
!side length = 1.0d0
ny_global = nx_global
nz_global = nx_global

lx = 1.0d0
ly = 1.0d0
lz = 1.0d0

x0 =-lx/2.0d0
y0 =-ly/2.0d0
z0 =-lz/2.0d0

else !TGV and KPV flows in periodic box of length 2pi
!a cartesian cube domain for TGV and KPV flows (
!side length = 2pi
ny_global = nx_global
nz_global = nx_global


lx = 2.0d0*pi
ly = 2.0d0*pi
lz = 2.0d0*pi

x0 = 0.0d0
y0 = 0.0d0
z0 = 0.0d0

end if

!where the domain is defined in R3 with equidistant cell size
!with uniform spatial step size
dx = lx/dfloat(nx_global)
dy = ly/dfloat(ny_global)
dz = lz/dfloat(nz_global)
      
!----------------------------------------------------------------------!
! Domain decomposition
! We only consider to decompose in z-direction (simplify)
!----------------------------------------------------------------------!

! Local array dimensions
  nx = nx_global
  ny = ny_global
  nz1 = int(nz_global/np)
  jip = nz_global - np*nz1 - 1
  
    !load balancing
	if(myid.le.jip) then
    nz=nz1+1
    else
    nz=nz1
    end if 

    
	if (nz.lt.3) then !because of +3/-3 stencil
    nx_global = nx_global*2
    goto 177
    end if

! Local grid  
! cell-centered grid points:
allocate(x(-2:nx+3,-2:ny+3,-2:nz+3))
allocate(y(-2:nx+3,-2:ny+3,-2:nz+3))
allocate(z(-2:nx+3,-2:ny+3,-2:nz+3))

!z_min= - 0.5d0*dz  + dfloat(myid)*dz*dfloat(nz)

if(myid.le.jip) then
z_min= - 0.5d0*dz  + dfloat(myid)*dz*dfloat(nz) 
else
z_min= - 0.5d0*dz  + dfloat(myid)*dz*dfloat(nz) + dfloat(jip+1)*dz  
end if


do k =-2, nz+3
do j =-2, ny+3
do i =-2, nx+3
    x(i,j,k) = x0 - 0.5d0*dx + dfloat(i)*dx 
    y(i,j,k) = y0 - 0.5d0*dy + dfloat(j)*dy     
    z(i,j,k) = z0 +    z_min + dfloat(k)*dz
end do
end do
end do

!total send and receive points
nsend = (nx+6)*(ny+6)

!----------------------------------------------------------------------!
!Problem definition and initialization
!----------------------------------------------------------------------!
!use 3 ghost cells within the domain (3 points overlap between domains)

!Allocate local arrays
allocate(q(-2:nx+3,-2:ny+3,-2:nz+3,5)) 	!primary array for conservative field variables
allocate(u(-2:nx+3,-2:ny+3,-2:nz+3,5)) 	!temporary array for conservative field variables
allocate(s(nx,ny,nz,5))           		!rhs array

!initial conditions
if (ipr.eq.3) then !KHI
	call initializeKHI(nx,ny,nz,x,y,z,q)
else if (ipr.eq.2) then !KPV
	call initializeKPV(nx,ny,nz,x,y,z,q)
else !TGV
	call initializeTGV(nx,ny,nz,x,y,z,q)
end if

!counting info  
if (cfl.ge.1.0d0) cfl=1.0d0
if (nsnap.lt.1) nsnap = 1 
iend = 0   
time = 0.0d0
ifile = 0
iout = 0
dtout = tmax/dfloat(nsnap)
tout = dtout 

if (idt.eq.0) then
	nt = int(tmax/dt) !number of time step for constant dt given by input
	if (mod(nt,nsnap).ne.0) nt = nt-mod(nt,nsnap)+nsnap
	dt = tmax/dfloat(nt)
	nf = nt/nsnap  
else if (idt.eq.1) then
!compute initial dt:

    	!Compute time step from cfl (since it is decaying only compute at initial time)
		call timestep(nx,ny,nz,dx,dy,dz,cfl,dt_local,q)

		!Compute dt within all processors (MPI_MIN)
    	call MPI_Reduce(dt_local, dt, 1, MPI_DOUBLE_PRECISION, &
                    MPI_MIN, 0, MPI_COMM_WORLD, ierr) 
 
		!Broadcast "dt" so that each processor will have the same value 
    	call MPI_Bcast(dt, 1, MPI_DOUBLE_PRECISION, &
                   0,  MPI_COMM_WORLD, ierr)


	nt = int(tmax/dt) !number of time step for constant dt given by input
	if (mod(nt,nsnap).ne.0) nt = nt-mod(nt,nsnap)+nsnap
	dt = tmax/dfloat(nt)
	nf = nt/nsnap
else
nt = 1000000000 !some big number (null number)
end if

!writing initial field (to make movie)
if (iwriteI.eq.1) then
call outputTec(nx,ny,nz,x,y,z,q,time,myid,ifile)
end if


!compute history (total energy = te )
call history(nx,ny,nz,q,te_local)

	! Compute the total energy within all processors (MPI_SUM)
    call MPI_Reduce(te_local, te, 1, MPI_DOUBLE_PRECISION, &
                    MPI_SUM, 0, MPI_COMM_WORLD, ierr) 
  
	! Broadcast "te" so that each processor will have the same value 
    call MPI_Bcast(te, 1, MPI_DOUBLE_PRECISION, &
                   0,  MPI_COMM_WORLD, ierr)   

	te = te/(nx_global*ny_global*nz_global)
	te0 = te
    
if (myid.eq.0) then
open(16,file='a_write.txt')
  
open(17,file='a_rate.plt')
write(17,*) 'variables ="t","e"'

open(18,file='a_energy.plt')
write(18,*) 'variables ="t","E"'
end if

if (myid.eq.0) write(18,*) time, te
  

!TVDRK3 coefficient
a = 1.0d0/3.0d0
b = 2.0d0/3.0d0

call cpu_time(t1)

!----------------------------------------------------------------------!
!Time integration
!----------------------------------------------------------------------!
do n=1,nt

    !------------------------------!
	!determine time step to advance
    !------------------------------!
    if (idt.eq.0) then  !constant time step (given input)

    	if (mod(n,nf).eq.0) iout = 1
        
	else if (idt.eq.1) then  !constant time step (given initial cfl)

    	if (mod(n,nf).eq.0) iout = 1
    
    else !compute adaptive time step from cfl

    	!Compute time step from cfl (since it is decaying only compute at initial time)
		call timestep(nx,ny,nz,dx,dy,dz,cfl,dt_local,q)

		!Compute dt within all processors (MPI_MIN)
    	call MPI_Reduce(dt_local, dt, 1, MPI_DOUBLE_PRECISION, &
                    MPI_MIN, 0, MPI_COMM_WORLD, ierr) 
 
		!Broadcast "dt" so that each processor will have the same value 
    	call MPI_Bcast(dt, 1, MPI_DOUBLE_PRECISION, &
                   0,  MPI_COMM_WORLD, ierr)
                    
    	!check for output file times
    	if((time+dt).ge.tout) then
    	dt = tout - time
    	tout=tout + dtout
    	iout=1
    	end if 

    	!check for final time step
    	if((time+dt).ge.tmax) then
    	dt = tmax-time
    	iend=1
    	end if
    
	end if   
    
    !Time counter
    time = time + dt
    

	!TVDRK3 Solver using MPI directives:

    !--------------------!
    !Step 1
    !--------------------!
	call rhs(nx,ny,nz,dx,dy,dz,q,s)
    
    !update
	do m=1,5
    do k=1,nz
    do j=1,ny
	do i=1,nx
    u(i,j,k,m) = q(i,j,k,m) + dt*s(i,j,k,m) 
    end do
    end do
    end do
    end do

    	!update periodic boundary conditions (global)
        !lateral bc
        call perbc_xy(nx,ny,nz,u)
        
		!horizontal bc
     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,5
        	ic = ic + 1
     		call MPI_Isend(u(-2,-2,nz,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(u(-2,-2,nz-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(u(-2,-2,nz-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Isend(u(-2,-2,1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(-2,-2,2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(-2,-2,3,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
        	call MPI_Irecv(u(-2,-2,0,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(-2,-2,-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(-2,-2,-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Irecv(u(-2,-2,nz+1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(-2,-2,nz+2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(-2,-2,nz+3,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    !SEND/RECEIVE DATA AMONG PROCESSORS
      
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
      	do m=1,5
        ic = ic + 1
     	call MPI_Isend(u(-2,-2,nz,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(u(-2,-2,nz-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(u(-2,-2,nz-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,5
        ic = ic + 1
       	call MPI_Isend(u(-2,-2,1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(-2,-2,2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(-2,-2,3,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
      	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,5
        ic = ic + 1
        call MPI_Irecv(u(-2,-2,0,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(-2,-2,-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(-2,-2,-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,5
        ic = ic + 1
       	call MPI_Irecv(u(-2,-2,nz+1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(-2,-2,nz+2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(-2,-2,nz+3,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do               
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)

   
    !--------------------!
	!Step 2
    !--------------------!
	call rhs(nx,ny,nz,dx,dy,dz,u,s)

	!update
	do m=1,5
    do k=1,nz
    do j=1,ny
	do i=1,nx
    u(i,j,k,m) = 0.75d0*q(i,j,k,m) + 0.25d0*u(i,j,k,m) + 0.25d0*dt*s(i,j,k,m) 
    end do
    end do
    end do
    end do

    	!update boundary conditions
        !lateral bc
        call perbc_xy(nx,ny,nz,u)
        
		!horizontal bc
        
     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then !periodic boundary conditions (global)
       
       		do m=1,5
        	ic = ic + 1
     		call MPI_Isend(u(-2,-2,nz,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(u(-2,-2,nz-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(u(-2,-2,nz-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Isend(u(-2,-2,1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(-2,-2,2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(-2,-2,3,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
        	call MPI_Irecv(u(-2,-2,0,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(-2,-2,-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(-2,-2,-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Irecv(u(-2,-2,nz+1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(-2,-2,nz+2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(-2,-2,nz+3,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
   	!SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
       	do m=1,5
        ic = ic + 1
     	call MPI_Isend(u(-2,-2,nz,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(u(-2,-2,nz-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(u(-2,-2,nz-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,5
        ic = ic + 1
       	call MPI_Isend(u(-2,-2,1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(-2,-2,2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(-2,-2,3,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,5
        ic = ic + 1
        call MPI_Irecv(u(-2,-2,0,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(-2,-2,-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(-2,-2,-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,5
        ic = ic + 1
       	call MPI_Irecv(u(-2,-2,nz+1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(-2,-2,nz+2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(-2,-2,nz+3,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
        end do               
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)

    
    !--------------------!
	!Step 3 
    !--------------------! 
	call rhs(nx,ny,nz,dx,dy,dz,u,s)
    
	!update
	do m=1,5
    do k=1,nz
    do j=1,ny
	do i=1,nx
    q(i,j,k,m) = a*q(i,j,k,m)+b*u(i,j,k,m)+b*dt*s(i,j,k,m) 
    end do
    end do
    end do 
    end do
   
    	!update boundary conditions
        !lateral bc
        call perbc_xy(nx,ny,nz,q)
        
		!horizontal bc
     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,5
        	ic = ic + 1
     		call MPI_Isend(q(-2,-2,nz,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(q(-2,-2,nz-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(q(-2,-2,nz-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Isend(q(-2,-2,1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(-2,-2,2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(-2,-2,3,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
        	call MPI_Irecv(q(-2,-2,0,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(-2,-2,-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(-2,-2,-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Irecv(q(-2,-2,nz+1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(-2,-2,nz+2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(-2,-2,nz+3,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    !SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
       	do m=1,5
        ic = ic + 1
     	call MPI_Isend(q(-2,-2,nz,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(q(-2,-2,nz-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(q(-2,-2,nz-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,5
        ic = ic + 1
       	call MPI_Isend(q(-2,-2,1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(q(-2,-2,2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(q(-2,-2,3,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,5
        ic = ic + 1
        call MPI_Irecv(q(-2,-2,0,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(q(-2,-2,-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(q(-2,-2,-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,5
        ic = ic + 1
       	call MPI_Irecv(q(-2,-2,nz+1,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(q(-2,-2,nz+2,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(q(-2,-2,nz+3,m), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do                
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)

	!----------------------------------------------------------------------!
    !Filtering (at the end of each time step)
    !----------------------------------------------------------------------!
	if (ifil.eq.2) then
		call filterSF7(nx,ny,nz,q)
	else if (ifil.eq.3) then
		call filterTam7(nx,ny,nz,q)
	end if 

	!Exchance filtered data among processors
	if (ifil.eq.2 .or. ifil.eq.3) then
        !update boundary conditions
        !lateral bc
        call perbc_xy(nx,ny,nz,q)
        
		!horizontal bc
     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,5
        	ic = ic + 1
     		call MPI_Isend(q(-2,-2,nz,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(q(-2,-2,nz-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(q(-2,-2,nz-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Isend(q(-2,-2,1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(-2,-2,2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(-2,-2,3,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,5
        	ic = ic + 1
        	call MPI_Irecv(q(-2,-2,0,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(-2,-2,-1,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(-2,-2,-2,m), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,5
        	ic = ic + 1
       		call MPI_Irecv(q(-2,-2,nz+1,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(-2,-2,nz+2,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(-2,-2,nz+3,m), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

	end if !end of filtering


    !----------------------------------------------------------------------!               
	!log file
    !----------------------------------------------------------------------!
    !if (myid.eq.0) write(*,*)n,time

    !----------------------------------------------------------------------!
    !I/O Files
    !----------------------------------------------------------------------!
    if (iout.eq.1) then
     
    ifile = ifile + 1

    !----------------------------------------------------------------------!
	! Wait until all processors come to this point before writing files
	!----------------------------------------------------------------------!
	call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    !write output for all processors in Tecplot format (not recommended)
    if (iwriteF.eq.1) then
	call outputTec(nx,ny,nz,x,y,z,q,time,myid,ifile)
    end if
    
	!write output in unformatted (recommended)
    if (iwriteS.eq.1) then 
	call output(nx,ny,nz,q,myid,ifile)
    end if	

    if (myid.eq.0) write(16,*)"solution file/time:",ifile,time

    iout = 0
    
    end if

    !----------------------------------------------------------------------!
	!Post-processing:
	!----------------------------------------------------------------------!
    
	!compute history (total energy = te)
	call history(nx,ny,nz,q,te_local)

    ! Compute the total energy within all processors (MPI_SUM)
    call MPI_Reduce(te_local, te, 1, MPI_DOUBLE_PRECISION, &
                    MPI_SUM, 0, MPI_COMM_WORLD, ierr) 
    
	! Broadcast "te" so that each processor will have the same value 
    call MPI_Bcast(te, 1, MPI_DOUBLE_PRECISION, &
                   0,  MPI_COMM_WORLD, ierr)

    te = te/(nx_global*ny_global*nz_global)
    
    if (myid.eq.0) write(18,*) time, te
      
    !compute dissipation rate
    rate = -(te-te0)/dt
    te0 = te
    
	if (myid.eq.0) write(17,*) time-dt*0.5d0, rate

   	!quit from the time loop if the desired maximum time reached
    if (iend.eq.1) goto 111
       
end do

111 continue
      
call cpu_time(t2)

if (myid.eq.0) close(16)
if (myid.eq.0) close(17)
if (myid.eq.0) close(18)
     
!write CPU data INCLUDE 
if (myid.eq.0) then
open(19,file='a_cpu.txt') 
write(19,*)"number of processor =", np
write(19,*)""
write(19,*)"local resolution    =", nx,ny,nz
write(19,*)"global resolution   =", nx_global,ny_global,nz_global
write(19,*)""
write(19,*)"cpu time  =", t2-t1, "  seconds"
write(19,*)"cpu time  =", (t2-t1)/60.0d0, "  minutes"
write(19,*)"cpu time  =", (t2-t1)/3600.0d0, "  hours" 
write(19,*)""
write(19,*)"final time step =", dt
write(19,*)"avarage time step =", tmax/dfloat(n)
write(19,*)"number of time integration points =", n
write(19,*)""
write(19,*)"cpu time/nt =", (t2-t1)/dfloat(n), " seconds"
write(19,*)"cpu time/nt =", (t2-t1)/dfloat(n)/60.0d0, " minutes"
write(19,*)"cpu time/nt=", (t2-t1)/dfloat(n)/3600.0d0, " hours"
write(19,*)""
write(19,*)"cpu time/nt/np =", (t2-t1)/dfloat(n)/dfloat(np), " seconds"
write(19,*)"cpu time/nt/np =", (t2-t1)/dfloat(n)/60.0d0/dfloat(np), " minutes"
write(19,*)"cpu time/nt/np =", (t2-t1)/dfloat(n)/3600.0d0/dfloat(np), " hours"
close(19)
end if

!----------------------------------------------------------------------!
! MPI final call
!----------------------------------------------------------------------!
call MPI_Finalize(ierr) 


end


!----------------------------------------------------------------------!
! Output files: ASCII
!----------------------------------------------------------------------!
subroutine output(nx,ny,nz,q,myid,ifile)
implicit none
integer::nx,ny,nz,myid,ifile
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
integer:: i,j,k,m
character(80):: charID, snapID, filename

write(charID,'(i5)') myid       !index for each processor 
write(snapID,'(i5)') ifile      !index for time snapshot


! Define the file name (write all conserved variables)
filename = "data_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.dat'
! Open the file and start writing the data
open(unit=19, file=filename)
do k = 0, nz
do j = 0, ny
do i = 0, nx
  	write(19,*) q(i,j,k,1),q(i,j,k,2),q(i,j,k,3),q(i,j,k,4),q(i,j,k,5)
end do
end do
end do
close(19)

! Write load information
! Define the file name
filename = "load_"// trim(adjustl(charID)) // '.dat'
open(unit=19, file=filename)
write(19,*)myid,nz
close(19)


end

!----------------------------------------------------------------------!
! Output files: binary
!----------------------------------------------------------------------!
subroutine output_binary(nx,ny,nz,q,myid,ifile)
implicit none
integer::nx,ny,nz,myid,ifile
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
integer:: i,j,k,m
character(80):: charID, snapID, filename

write(charID,'(i5)') myid       !index for each processor 
write(snapID,'(i5)') ifile      !index for time snapshot


! Define the file name (write all conserved variables)
filename = "data_binary_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.dat'
! Open the file and start writing the data
open(unit=19, file=filename, form='unformatted')
do k = 0, nz
do j = 0, ny
do i = 0, nx
  	write(19) q(i,j,k,1),q(i,j,k,2),q(i,j,k,3),q(i,j,k,4),q(i,j,k,5)
end do
end do
end do
close(19)

! Write load information
! Define the file name
filename = "load_"// trim(adjustl(charID)) // '.dat'
open(unit=19, file=filename)
write(19,*)myid,nz
close(19)


end

!----------------------------------------------------------------------!
! Output files: Tecplot
!----------------------------------------------------------------------!
subroutine outputTec(nx,ny,nz,x,y,z,q,time,myid,ifile)
implicit none
integer::nx,ny,nz,myid,ifile
real*8 ::time,p,gamma
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8 ::x(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::y(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::z(-2:nx+3,-2:ny+3,-2:nz+3)
integer:: i,j,k
character(80):: charID, snapID, timeID, filename

common /fluids/ gamma

write(charID,'(i5)') myid       !index for each processor 
write(snapID,'(i5)') ifile      !index for time snapshot
write(timeID,'(f10.2)') time    !index for plotting time

! Define the file name
filename = "density_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.plt'

! Open the file and start writing the data
open(unit=19, file=filename)

! Tecplot header
write(19,*) 'title =', '"Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID))// &
            '_Time_'// trim(adjustl(timeID)), '"'
write(19,*) 'variables = "x", "y", "z", "r"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID)), &
            ' i=', nx+1, ' j=', ny+1, ' k=', nz+1, ' f=point'

! Write density data
do k = 0, nz
do j = 0, ny
do i = 0, nx
  	write(19,*) x(i,j,k), y(i,j,k), z(i,j,k), q(i,j,k,1)
end do 
end do
end do

close(19)


! Define the file name
filename = "pressure_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.plt'

! Open the file and start writing the data
open(unit=19, file=filename)

! Tecplot header
write(19,*) 'title =', '"Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID))// &
            '_Time_'// trim(adjustl(timeID)), '"'
write(19,*) 'variables = "x", "y", "z", "p"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID)), &
            ' i=', nx+1, ' j=', ny+1, ' k=', nz+1, ' f=point'

! Write density data
do k = 0, nz
do j = 0, ny
do i = 0, nx
p = (gamma-1.0d0)*(q(i,j,k,5)-0.5d0*(q(i,j,k,2)*q(i,j,k,2)/q(i,j,k,1) &
                             +q(i,j,k,3)*q(i,j,k,3)/q(i,j,k,1) &
                             +q(i,j,k,4)*q(i,j,k,4)/q(i,j,k,1) ))
                                    
write(19,*) x(i,j,k), y(i,j,k), z(i,j,k), p
end do 
end do
end do

close(19)


! Define the file name
filename = "vel_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.plt'

! Open the file and start writing the data
open(unit=19, file=filename)

! Tecplot header
write(19,*) 'title =', '"Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID))// &
            '_Time_'// trim(adjustl(timeID)), '"'
write(19,*) 'variables = "x", "y", "z", "u", "v", "w"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID)), &
            ' i=', nx+1, ' j=', ny+1,' k=', nz+1,  ' f=point'

! Write velocity data
do k = 0, nz
do j = 0, ny
do i = 0, nx
  	write(19,*) x(i,j,k), y(i,j,k), z(i,j,k), q(i,j,k,2)/q(i,j,k,1), &
    q(i,j,k,3)/q(i,j,k,1), q(i,j,k,4)/q(i,j,k,1)
end do 
end do
end do

close(19)


! Write load information
! Define the file name
filename = "load_"// trim(adjustl(charID)) // '.dat'
open(unit=19, file=filename)
write(19,*)myid,nz
close(19)


end


!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!TGV: Taylor Green Vortex
!-----------------------------------------------------------------------------------!
subroutine initializeTGV(nx,ny,nz,x,y,z,q)
implicit none
integer::nx,ny,nz,i,j,k
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8 ::x(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::y(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::z(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::r,u,v,w,p,e
real*8 ::gamma,vinit,rinit,pinit,tinit
real*8 ::xbl,ybl,zbl,ma

common /fluids/ gamma
common /mach/ ma

tinit = 1.0d0
rinit = 1.0d0
vinit = 1.0d0
pinit = rinit*tinit/(gamma*ma*ma)

 
do k=-2,nz+3
do j=-2,ny+3
do i=-2,nx+3 
  
    xbl = x(i,j,k)
    ybl = y(i,j,k)
    zbl = z(i,j,k)
	
    r = rinit
    u = vinit*dsin(xbl)*dcos(ybl)*dcos(zbl)
    v =-vinit*dcos(xbl)*dsin(ybl)*dcos(zbl)
    w = 0.0d0
    p = pinit + rinit*(vinit**2)/(16.0d0)*(dcos(2.0d0*xbl)+dcos(2.0d0*ybl))*(dcos(2.0d0*zbl)+2.0d0)
    
    e = p/(r*(gamma-1.0d0))+0.5d0*(u*u+v*v+w*w)

      
    !construct conservative variables     
	q(i,j,k,1)=r
	q(i,j,k,2)=r*u
    q(i,j,k,3)=r*v
	q(i,j,k,4)=r*w
   	q(i,j,k,5)=r*e 
       
end do
end do
end do


return
end

!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!KPV: Kida-Pelz Vortex
!Pressure for initial conditions are coppied for TGV, needs to be changed.
!-----------------------------------------------------------------------------------!
subroutine initializeKPV(nx,ny,nz,x,y,z,q)
implicit none
integer::nx,ny,nz,i,j,k
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8 ::x(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::y(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::z(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::r,u,v,w,p,e
real*8 ::gamma,vinit,rinit,pinit,tinit
real*8 ::xbl,ybl,zbl,ma

common /fluids/ gamma
common /mach/ ma

tinit = 1.0d0
rinit = 1.0d0
vinit = 1.0d0
pinit = rinit*tinit/(gamma*ma*ma)

 
do k=-2,nz+3
do j=-2,ny+3
do i=-2,nx+3 
  
    xbl = x(i,j,k)
    ybl = y(i,j,k)
    zbl = z(i,j,k)
	
    r = rinit
    u = vinit*dsin(xbl)*(dcos(3.0d0*ybl)*dcos(zbl)-dcos(ybl)*dcos(3.0d0*zbl))
    v = vinit*dsin(ybl)*(dcos(3.0d0*zbl)*dcos(xbl)-dcos(zbl)*dcos(3.0d0*xbl))
    w = vinit*dsin(zbl)*(dcos(3.0d0*xbl)*dcos(ybl)-dcos(xbl)*dcos(3.0d0*ybl))
    !p ????
    p = pinit + rinit*(vinit**2)/(16.0d0)*(dcos(2.0d0*xbl)+dcos(2.0d0*ybl))*(dcos(2.0d0*zbl)+2.0d0)
    
    e = p/(r*(gamma-1.0d0))+0.5d0*(u*u+v*v+w*w)

      
    !construct conservative variables     
	q(i,j,k,1)=r
	q(i,j,k,2)=r*u
    q(i,j,k,3)=r*v
	q(i,j,k,4)=r*w
   	q(i,j,k,5)=r*e 
       
end do
end do
end do


return
end


!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!KHI: Kelvin-Helmholtz Instability
!-----------------------------------------------------------------------------------!
subroutine initializeKHI(nx,ny,nz,x,y,z,q)
implicit none
integer::nx,ny,nz,i,j,k
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8 ::x(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::y(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::z(-2:nx+3,-2:ny+3,-2:nz+3)
real*8 ::r,u,v,w,p,e
real*8 ::gamma,pi,ma,aa
real*8 ::xbl,ybl,zbl
real*8 ::ukhi,denr,ur,alpha,ampy

common /fluids/ gamma
common /mach/ ma
common /KHIconstants/ ukhi,denr,alpha,ampy

pi = 4.0d0*datan(1.0d0)
ur = 1.0d0

 
do k=-2,nz+3
do j=-2,ny+3
do i=-2,nx+3 
  
    xbl = x(i,j,k)
    ybl = y(i,j,k)
    zbl = z(i,j,k)

  	if(dabs(zbl).ge.0.25d0) then !outer region     
    	r = 1.0d0
		u = ukhi
    	v = ampy*dsin(2.0d0*pi*alpha*ybl)
        w = ampy*dsin(2.0d0*pi*alpha*xbl) 
    	p = 2.5d0          
    else !inner region
        r = denr
		u =-ukhi*ur
    	v = ampy*dsin(2.0d0*pi*alpha*ybl)
        w = ampy*dsin(2.0d0*pi*alpha*xbl)
    	p = 2.5d0       
    end if
    	
    
    e = p/(r*(gamma-1.0d0))+0.5d0*(u*u+v*v+w*w)

      
    !construct conservative variables     
	q(i,j,k,1)=r
	q(i,j,k,2)=r*u
    q(i,j,k,3)=r*v
	q(i,j,k,4)=r*w
   	q(i,j,k,5)=r*e 
       
end do
end do
end do

!compute reference mach number based on outer region
ma = ukhi/dsqrt(gamma*p/1.0d0) 

open(199,file='a_initial_Mach_KHI.txt') 
write(199,*)"initial mach number (outer region)=", ma
write(199,*)"initial mach number (inner region)=", ukhi/dsqrt(gamma*p/denr) 
close(199)

return
end

!-----------------------------------------------------------------------------------!
!Periodic boundary conditions in x-y directions
!-----------------------------------------------------------------------------------!
subroutine perbc_xy(nx,ny,nz,q)
implicit none
integer::nx,ny,nz,i,j,k,m
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)


do m=1,5
  
	do k=1,nz
	do i=1,nx					!Ask question?
	q(i,0,k,m)   =q(i,ny,k,m)	!front of the domain
    q(i,-1,k,m)  =q(i,ny-1,k,m)	!front of the domain
    q(i,-2,k,m)  =q(i,ny-2,k,m)	!front of the domain
    
	q(i,ny+1,k,m)=q(i,1,k,m) 	!back of the domain	
	q(i,ny+2,k,m)=q(i,2,k,m)	!back of the domain	
	q(i,ny+3,k,m)=q(i,3,k,m)	!back of the domain
	end do
    end do
     
    do k=1,nz
	do j=-2,ny+3
	q( 0,j,k,m)  =q(nx,j,k,m)	!left of the domain
    q(-1,j,k,m)  =q(nx-1,j,k,m)	!left of the domain
    q(-2,j,k,m)  =q(nx-2,j,k,m)	!left of the domain
    
	q(nx+1,j,k,m)=q(1,j,k,m) 	!right of the domain	
	q(nx+2,j,k,m)=q(2,j,k,m)	!right of the domain	
	q(nx+3,j,k,m)=q(3,j,k,m)	!right of the domain
	end do
	end do

end do

return
end


!-----------------------------------------------------------------------------------!
!Filtering (SF-7)
!-----------------------------------------------------------------------------------!
subroutine filterSF7(nx,ny,nz,q)
implicit none
integer::nx,ny,nz,i,j,k,m
real*8 ::sigma,d0,d1,d2,d3
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8,allocatable:: u(:,:,:,:),v(:,:,:,:),w(:,:,:,:)

common /filterstrength/ sigma


allocate(u(-2:nx+3,-2:ny+3,-2:nz+3,5))
allocate(v(-2:nx+3,-2:ny+3,-2:nz+3,5))
allocate(w(-2:nx+3,-2:ny+3,-2:nz+3,5))

do m=1,5
do k=-2,nz+3  
do j=-2,ny+3
do i=-2,nx+3
u(i,j,k,m) = q(i,j,k,m)
end do
end do
end do
end do

d0 = 5.0d0/16.0d0
d1 =-15.0d0/64.0d0
d2 = 3.0d0/32.0d0
d3 =-1.0d0/64.0d0

!do m=1,5
!do k=1,nz  
!do j=1,ny
!do i=1,nx
!q(i,j,k,m) = u(i,j,k,m) - sigma*(3.0d0*d0*u(i,j,k,m) &
!                    + d1*(u(i+1,j,k,m)+u(i-1,j,k,m)+u(i,j+1,k,m)+u(i,j-1,k,m)+u(i,j,k+1,m)+u(i,j,k-1,m)) &
!                    + d2*(u(i+2,j,k,m)+u(i-2,j,k,m)+u(i,j+2,k,m)+u(i,j-2,k,m)+u(i,j,k+2,m)+u(i,j,k-2,m)) &
!                    + d3*(u(i+3,j,k,m)+u(i-3,j,k,m)+u(i,j+3,k,m)+u(i,j-3,k,m)+u(i,j,k+3,m)+u(i,j,k-3,m)) )                               
!end do
!end do
!end do
!end do

!filter in z
do m=1,5
do k=1,nz
do j=1,ny
do i=1,nx
v(i,j,k,m) = u(i,j,k,m) - sigma*(d0*u(i,j,k,m) &
                           + d1*(u(i,j,k+1,m)+u(i,j,k-1,m)) &
                           + d2*(u(i,j,k+2,m)+u(i,j,k-2,m)) &
                           + d3*(u(i,j,k+3,m)+u(i,j,k-3,m)) )
end do
end do
end do
end do

call perbc_xy(nx,ny,nz,v)

!filter in y
do m=1,5
do k=1,nz
do j=1,ny
do i=1,nx
w(i,j,k,m) = v(i,j,k,m) - sigma*(d0*v(i,j,k,m) &
                           + d1*(v(i,j+1,k,m)+v(i,j-1,k,m)) &
                           + d2*(v(i,j+2,k,m)+v(i,j-2,k,m)) &
                           + d3*(v(i,j+3,k,m)+v(i,j-3,k,m)) ) 
end do
end do
end do
end do  

call perbc_xy(nx,ny,nz,w)

!filter in x
do m=1,5
do k=1,nz
do j=1,ny
do i=1,nx
q(i,j,k,m) = w(i,j,k,m) - sigma*(d0*w(i,j,k,m) &
                           + d1*(w(i+1,j,k,m)+w(i-1,j,k,m)) &
                           + d2*(w(i+2,j,k,m)+w(i-2,j,k,m)) &
                           + d3*(w(i+3,j,k,m)+w(i-3,j,k,m)) )                               
end do
end do
end do
end do

call perbc_xy(nx,ny,nz,q)

deallocate(u,v,w)

return
end


!-----------------------------------------------------------------------------------!
!Filtering (Tam-7)
!-----------------------------------------------------------------------------------!
subroutine filterTam7(nx,ny,nz,q)
implicit none
integer::nx,ny,nz,i,j,k,m
real*8 ::sigma,d0,d1,d2,d3
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8,allocatable:: u(:,:,:,:),v(:,:,:,:),w(:,:,:,:)

common /filterstrength/ sigma

allocate(u(-2:nx+3,-2:ny+3,-2:nz+3,5))
allocate(v(-2:nx+3,-2:ny+3,-2:nz+3,5))
allocate(w(-2:nx+3,-2:ny+3,-2:nz+3,5))

do m=1,5
do k=-2,nz+3  
do j=-2,ny+3
do i=-2,nx+3
u(i,j,k,m) = q(i,j,k,m)
end do
end do
end do
end do

d0 = 0.287392842460d0
d1 =-0.226146951809d0
d2 = 0.106303578770d0
d3 =-0.023853048191d0

!do m=1,5
!do k=1,nz  
!do j=1,ny
!do i=1,nx
!q(i,j,k,m) = u(i,j,k,m) - sigma*(3.0d0*d0*u(i,j,k,m) &
!                    + d1*(u(i+1,j,k,m)+u(i-1,j,k,m)+u(i,j+1,k,m)+u(i,j-1,k,m)+u(i,j,k+1,m)+u(i,j,k-1,m)) &
!                    + d2*(u(i+2,j,k,m)+u(i-2,j,k,m)+u(i,j+2,k,m)+u(i,j-2,k,m)+u(i,j,k+2,m)+u(i,j,k-2,m)) &
!                    + d3*(u(i+3,j,k,m)+u(i-3,j,k,m)+u(i,j+3,k,m)+u(i,j-3,k,m)+u(i,j,k+3,m)+u(i,j,k-3,m)) )                             
!end do
!end do
!end do
!end do

!filter in z
do m=1,5
do k=1,nz
do j=1,ny
do i=1,nx
v(i,j,k,m) = u(i,j,k,m) - sigma*(d0*u(i,j,k,m) &
                           + d1*(u(i,j,k+1,m)+u(i,j,k-1,m)) &
                           + d2*(u(i,j,k+2,m)+u(i,j,k-2,m)) &
                           + d3*(u(i,j,k+3,m)+u(i,j,k-3,m)) )
end do
end do
end do
end do

call perbc_xy(nx,ny,nz,v)

!filter in y
do m=1,5
do k=1,nz
do j=1,ny
do i=1,nx
w(i,j,k,m) = v(i,j,k,m) - sigma*(d0*v(i,j,k,m) &
                           + d1*(v(i,j+1,k,m)+v(i,j-1,k,m)) &
                           + d2*(v(i,j+2,k,m)+v(i,j-2,k,m)) &
                           + d3*(v(i,j+3,k,m)+v(i,j-3,k,m)) ) 
end do
end do
end do
end do  

call perbc_xy(nx,ny,nz,w)

!filter in x
do m=1,5
do k=1,nz
do j=1,ny
do i=1,nx
q(i,j,k,m) = w(i,j,k,m) - sigma*(d0*w(i,j,k,m) &
                           + d1*(w(i+1,j,k,m)+w(i-1,j,k,m)) &
                           + d2*(w(i+2,j,k,m)+w(i-2,j,k,m)) &
                           + d3*(w(i+3,j,k,m)+w(i-3,j,k,m)) )                               
end do
end do
end do
end do

call perbc_xy(nx,ny,nz,q)


deallocate(u,v,w)

return
end

!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS)
!-----------------------------------------------------------------------------------!
subroutine rhs(nx,ny,nz,dx,dy,dz,q,s)
implicit none
integer::nx,ny,nz,isgs,imodel
real*8 ::dx,dy,dz
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5),s(nx,ny,nz,5)

common /modeling/ imodel,isgs


!Compute RHS
if (imodel.eq.1) then 
  	!Euler Model
	if (isgs.eq.1) then !WENO for inviscid term
    call rhsILES(nx,ny,nz,dx,dy,dz,q,s)
    else !Central for inviscid term
    call rhsCS(nx,ny,nz,dx,dy,dz,q,s)
    end if
else 
  	!NS Model
  	!Viscous terms are always central difference
	if (isgs.eq.1) then !WENO for inviscid term
    call rhsILES(nx,ny,nz,dx,dy,dz,q,s)
    call rhsVIS(nx,ny,nz,dx,dy,dz,q,s)
    else !Central for inviscid term
    call rhsCS(nx,ny,nz,dx,dy,dz,q,s)
    call rhsVIS(nx,ny,nz,dx,dy,dz,q,s)	
    end if
end if


return 
end 

!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS) for viscous term (central) finite volume
!Sixth order interpolations for all terms - recommended
!-----------------------------------------------------------------------------------!
subroutine rhsVIS(nx,ny,nz,dx,dy,dz,q,s)
implicit none
integer::nx,ny,nz,i,j,k,m !,imu
real*8 ::dx,dy,dz,re,pr,gamma,Tref,ma,mux,muy,muz
real*8 ::g,a,b,c,dx1,dx3,dx5,dy1,dy3,dy5,dz1,dz3,dz5
real*8 ::gi,ai,bi,ci,gd,ad,bd,cd,dx2,dx4,dx6,dy2,dy4,dy6,dz2,dz4,dz6
real*8 ::g1,g2,g3,g4,cc,u,v,w,txx,tyy,tzz,txy,txz,tyz,tx,ty,tz
real*8 ::ux,uy,uz,vx,vy,vz,wx,wy,wz,qx,qy,qz
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8 ::s(-2:nx+3,-2:ny+3,-2:nz+3,5)
real*8,allocatable:: t(:,:,:),vf(:,:,:,:),vg(:,:,:,:),vh(:,:,:,:),mu(:,:,:)
real*8,allocatable:: cd1(:,:,:),cd2(:,:,:),cd3(:,:,:),cd4(:,:,:)

common /viscosity/ re,pr,Tref
common /fluids/ gamma
common /mach/ ma
!common /Sutherland/ imu

allocate(t(-2:nx+3,-2:ny+3,-2:nz+3),mu(-2:nx+3,-2:ny+3,-2:nz+3))
allocate(vf(0:nx,1:ny,1:nz,5),vg(1:nx,0:ny,1:nz,5),vh(1:nx,1:ny,0:nz,5))

!call perbc_xy(nx,ny,nz,q)

!for derivatives (finite volume)
g = 1.0d0/180.0d0
a = 245.0d0
b =-75.0d0
c = 10.0d0

dx1 = 1.0d0/dx
dx3 = 1.0d0/(3.0d0*dx)
dx5 = 1.0d0/(5.0d0*dx)

dy1 = 1.0d0/dy
dy3 = 1.0d0/(3.0d0*dy)
dy5 = 1.0d0/(5.0d0*dy)

dz1 = 1.0d0/dz
dz3 = 1.0d0/(3.0d0*dz)
dz5 = 1.0d0/(5.0d0*dz)

!for interpolation (finite volume)
gi = 1.0d0/60.0d0
ai = 37.0d0
bi =-8.0d0
ci = 1.0d0

!for cross derivatives
gd = 1.0d0/10.0d0
ad = 15.0d0
bd =-6.0d0
cd = 1.0d0


dx2 = 1.0d0/(2.0d0*dx)
dx4 = 1.0d0/(4.0d0*dx)
dx6 = 1.0d0/(6.0d0*dx)

dy2 = 1.0d0/(2.0d0*dy)
dy4 = 1.0d0/(4.0d0*dy)
dy6 = 1.0d0/(6.0d0*dy)

dz2 = 1.0d0/(2.0d0*dz)
dz4 = 1.0d0/(4.0d0*dz)
dz6 = 1.0d0/(6.0d0*dz)

g1 = 1.0d0/re
g2 = (2.0d0/3.0d0)/re
g3 =-1.0d0/(re*pr*ma*ma*(gamma-1.0d0))
g4 = (gamma-1.0d0)*gamma*ma*ma

cc = 110.4d0/Tref

!compute temperature
do k=-2,nz+3
do j=-2,ny+3
do i=-2,nx+3
t(i,j,k) = g4*(q(i,j,k,5)/q(i,j,k,1) &
       - 0.5d0*((q(i,j,k,2)/q(i,j,k,1))*(q(i,j,k,2)/q(i,j,k,1)) &
               +(q(i,j,k,3)/q(i,j,k,1))*(q(i,j,k,3)/q(i,j,k,1)) &
               +(q(i,j,k,4)/q(i,j,k,1))*(q(i,j,k,4)/q(i,j,k,1)) ) )
               
mu(i,j,k) = (t(i,j,k)**(1.5))*(1.0d0+cc)/(t(i,j,k)+cc) !Sutherland's Law
               
end do
end do
end do

!-------------------------!
!In x-direction:
!-------------------------!
allocate(cd1(-2:nx+3,1:ny,1:nz),cd2(-2:nx+3,1:ny,1:nz))
allocate(cd3(-2:nx+3,1:ny,1:nz),cd4(-2:nx+3,1:ny,1:nz))
!compute cross-derivatives: uy,uz,vy,wz
do k=1,nz
do j=1,ny
do i=-2,nx+3

!uy
cd1(i,j,k) = gd*(ad*dy2*( q(i,j+1,k,2)/q(i,j+1,k,1) - q(i,j-1,k,2)/q(i,j-1,k,1) ) &
               + bd*dy4*( q(i,j+2,k,2)/q(i,j+2,k,1) - q(i,j-2,k,2)/q(i,j-2,k,1) ) &
               + cd*dy6*( q(i,j+3,k,2)/q(i,j+3,k,1) - q(i,j-3,k,2)/q(i,j-3,k,1) ) )

!uz
cd2(i,j,k) = gd*(ad*dz2*( q(i,j,k+1,2)/q(i,j,k+1,1) - q(i,j,k-1,2)/q(i,j,k-1,1) ) &
               + bd*dz4*( q(i,j,k+2,2)/q(i,j,k+2,1) - q(i,j,k-2,2)/q(i,j,k-2,1) ) &
               + cd*dz6*( q(i,j,k+3,2)/q(i,j,k+3,1) - q(i,j,k-3,2)/q(i,j,k-3,1) ) )


!vy
cd3(i,j,k) = gd*(ad*dy2*( q(i,j+1,k,3)/q(i,j+1,k,1) - q(i,j-1,k,3)/q(i,j-1,k,1) ) &
               + bd*dy4*( q(i,j+2,k,3)/q(i,j+2,k,1) - q(i,j-2,k,3)/q(i,j-2,k,1) ) &
               + cd*dy6*( q(i,j+3,k,3)/q(i,j+3,k,1) - q(i,j-3,k,3)/q(i,j-3,k,1) ) )

!wz      
cd4(i,j,k) = gd*(ad*dz2*( q(i,j,k+1,4)/q(i,j,k+1,1) - q(i,j,k-1,4)/q(i,j,k-1,1) ) &
               + bd*dz4*( q(i,j,k+2,4)/q(i,j,k+2,1) - q(i,j,k-2,4)/q(i,j,k-2,1) ) &
               + cd*dz6*( q(i,j,k+3,4)/q(i,j,k+3,1) - q(i,j,k-3,4)/q(i,j,k-3,1) ) )
      
end do
end do
end do
      
!compute viscous fluxes at cell interfaces:

do k=1,nz
do j=1,ny
do i=0,nx

ux = g*(  a*dx1*( q(i+1,j,k,2)/q(i+1,j,k,1) - q(i,j,k,2)/q(i,j,k,1) ) &
        + b*dx3*( q(i+2,j,k,2)/q(i+2,j,k,1) - q(i-1,j,k,2)/q(i-1,j,k,1) ) &
        + c*dx5*( q(i+3,j,k,2)/q(i+3,j,k,1) - q(i-2,j,k,2)/q(i-2,j,k,1) ) )

vx = g*(a*dx1*( q(i+1,j,k,3)/q(i+1,j,k,1) - q(i,j,k,3)/q(i,j,k,1) ) &
      + b*dx3*( q(i+2,j,k,3)/q(i+2,j,k,1) - q(i-1,j,k,3)/q(i-1,j,k,1) ) &
      + c*dx5*( q(i+3,j,k,3)/q(i+3,j,k,1) - q(i-2,j,k,3)/q(i-2,j,k,1) ) )


wx = g*(a*dx1*( q(i+1,j,k,4)/q(i+1,j,k,1) - q(i,j,k,4)/q(i,j,k,1) ) &
      + b*dx3*( q(i+2,j,k,4)/q(i+2,j,k,1) - q(i-1,j,k,4)/q(i-1,j,k,1) ) &
      + c*dx5*( q(i+3,j,k,4)/q(i+3,j,k,1) - q(i-2,j,k,4)/q(i-2,j,k,1) ) )

uy = gi*(ai*( cd1(i+1,j,k) + cd1(i,j,k) ) &
       + bi*( cd1(i+2,j,k) + cd1(i-1,j,k) ) &
       + ci*( cd1(i+3,j,k) + cd1(i-2,j,k) ) )

uz = gi*(ai*( cd2(i+1,j,k) + cd2(i,j,k) ) &
       + bi*( cd2(i+2,j,k) + cd2(i-1,j,k) ) &
       + ci*( cd2(i+3,j,k) + cd2(i-2,j,k) ) )

vy = gi*(ai*( cd3(i+1,j,k) + cd3(i,j,k) ) &
       + bi*( cd3(i+2,j,k) + cd3(i-1,j,k) ) &
       + ci*( cd3(i+3,j,k) + cd3(i-2,j,k) ) )

wz = gi*(ai*( cd4(i+1,j,k) + cd4(i,j,k) ) &
       + bi*( cd4(i+2,j,k) + cd4(i-1,j,k) ) &
       + ci*( cd4(i+3,j,k) + cd4(i-2,j,k) ) )
  
tx = g*(a*dx1*( t(i+1,j,k) - t(i,j,k) ) &
      + b*dx3*( t(i+2,j,k) - t(i-1,j,k) ) &
      + c*dx5*( t(i+3,j,k) - t(i-2,j,k) ) )


u  = gi*(ai*( q(i+1,j,k,2)/q(i+1,j,k,1) + q(i,j,k,2)/q(i,j,k,1) ) &
       + bi*( q(i+2,j,k,2)/q(i+2,j,k,1) + q(i-1,j,k,2)/q(i-1,j,k,1) ) &
       + ci*( q(i+3,j,k,2)/q(i+3,j,k,1) + q(i-2,j,k,2)/q(i-2,j,k,1) ) )
      
v  = gi*(ai*( q(i+1,j,k,3)/q(i+1,j,k,1) + q(i,j,k,3)/q(i,j,k,1) ) &
       + bi*( q(i+2,j,k,3)/q(i+2,j,k,1) + q(i-1,j,k,3)/q(i-1,j,k,1) ) &
       + ci*( q(i+3,j,k,3)/q(i+3,j,k,1) + q(i-2,j,k,3)/q(i-2,j,k,1) ) )

w  = gi*(ai*( q(i+1,j,k,4)/q(i+1,j,k,1) + q(i,j,k,4)/q(i,j,k,1) ) &
       + bi*( q(i+2,j,k,4)/q(i+2,j,k,1) + q(i-1,j,k,4)/q(i-1,j,k,1) ) &
       + ci*( q(i+3,j,k,4)/q(i+3,j,k,1) + q(i-2,j,k,4)/q(i-2,j,k,1) ) )       

mux = gi*(ai*( mu(i+1,j,k) + mu(i,j,k) ) &
        + bi*( mu(i+2,j,k) + mu(i-1,j,k) ) &
        + ci*( mu(i+3,j,k) + mu(i-2,j,k) ) )


txx = g2*mux*(2.0d0*ux - vy - wz)
txy = g1*mux*(uy + vx)
txz = g1*mux*(uz + wx) 
qx  = g3*mux*tx

vf(i,j,k,1) = 0.0d0
vf(i,j,k,2) = txx
vf(i,j,k,3) = txy
vf(i,j,k,4) = txz
vf(i,j,k,5) = u*txx + v*txy + w*txz - qx


end do
end do
end do

! compute RHS contribution due to viscous term (central difference & finite volume)
do m=1,5
do k=1,nz  
do j=1,ny
do i=1,nx
s(i,j,k,m) = s(i,j,k,m) + (vf(i,j,k,m)-vf(i-1,j,k,m))/dx                        
end do
end do
end do
end do

deallocate(cd1,cd2,cd3,cd4)

!-------------------------!
!In y-direction:
!-------------------------!
allocate(cd1(1:nx,-2:ny+3,1:nz),cd2(1:nx,-2:ny+3,1:nz))
allocate(cd3(1:nx,-2:ny+3,1:nz),cd4(1:nx,-2:ny+3,1:nz))
!compute cross-derivatives: vx,vz,ux,wz
do k=1,nz
do j=-2,ny+3
do i=1,nx

!vx
cd1(i,j,k) = gd*(ad*dx2*( q(i+1,j,k,3)/q(i+1,j,k,1) - q(i-1,j,k,3)/q(i-1,j,k,1) ) &
               + bd*dx4*( q(i+2,j,k,3)/q(i+2,j,k,1) - q(i-2,j,k,3)/q(i-2,j,k,1) ) &
               + cd*dx6*( q(i+3,j,k,3)/q(i+3,j,k,1) - q(i-3,j,k,3)/q(i-3,j,k,1) ) )

!vz
cd2(i,j,k) = gd*(ad*dz2*( q(i,j,k+1,3)/q(i,j,k+1,1) - q(i,j,k-1,3)/q(i,j,k-1,1) ) &
               + bd*dz4*( q(i,j,k+2,3)/q(i,j,k+2,1) - q(i,j,k-2,3)/q(i,j,k-2,1) ) &
               + cd*dz6*( q(i,j,k+3,3)/q(i,j,k+3,1) - q(i,j,k-3,3)/q(i,j,k-3,1) ) )


!ux
cd3(i,j,k) = gd*(ad*dx2*( q(i+1,j,k,2)/q(i+1,j,k,1) - q(i-1,j,k,2)/q(i-1,j,k,1) ) &
               + bd*dx4*( q(i+2,j,k,2)/q(i+2,j,k,1) - q(i-2,j,k,2)/q(i-2,j,k,1) ) &
               + cd*dx6*( q(i+3,j,k,2)/q(i+3,j,k,1) - q(i-3,j,k,2)/q(i-3,j,k,1) ) )

!wz      
cd4(i,j,k) = gd*(ad*dz2*( q(i,j,k+1,4)/q(i,j,k+1,1) - q(i,j,k-1,4)/q(i,j,k-1,1) ) &
               + bd*dz4*( q(i,j,k+2,4)/q(i,j,k+2,1) - q(i,j,k-2,4)/q(i,j,k-2,1) ) &
               + cd*dz6*( q(i,j,k+3,4)/q(i,j,k+3,1) - q(i,j,k-3,4)/q(i,j,k-3,1) ) )
      
end do
end do
end do
      
!compute viscous fluxes at cell interfaces:

do k=1,nz
do j=0,ny
do i=1,nx

uy = g*(a*dy1*( q(i,j+1,k,2)/q(i,j+1,k,1) - q(i,j,k,2)/q(i,j,k,1) ) &
      + b*dy3*( q(i,j+2,k,2)/q(i,j+2,k,1) - q(i,j-1,k,2)/q(i,j-1,k,1) ) &
      + c*dy5*( q(i,j+3,k,2)/q(i,j+3,k,1) - q(i,j-2,k,2)/q(i,j-2,k,1) ) )

vy = g*(a*dy1*( q(i,j+1,k,3)/q(i,j+1,k,1) - q(i,j,k,3)/q(i,j,k,1) ) &
      + b*dy3*( q(i,j+2,k,3)/q(i,j+2,k,1) - q(i,j-1,k,3)/q(i,j-1,k,1) ) &
      + c*dy5*( q(i,j+3,k,3)/q(i,j+3,k,1) - q(i,j-2,k,3)/q(i,j-2,k,1) ) )
      
wy = g*(a*dy1*( q(i,j+1,k,4)/q(i,j+1,k,1) - q(i,j,k,4)/q(i,j,k,1) ) &
      + b*dy3*( q(i,j+2,k,4)/q(i,j+2,k,1) - q(i,j-1,k,4)/q(i,j-1,k,1) ) &
      + c*dy5*( q(i,j+3,k,4)/q(i,j+3,k,1) - q(i,j-2,k,4)/q(i,j-2,k,1) ) )
      

vx = gi*(ai*( cd1(i,j+1,k) + cd1(i,j,k) ) &
       + bi*( cd1(i,j+2,k) + cd1(i,j-1,k) ) &
       + ci*( cd1(i,j+3,k) + cd1(i,j-2,k) ) )

vz = gi*(ai*( cd2(i,j+1,k) + cd2(i,j,k) ) &
       + bi*( cd2(i,j+2,k) + cd2(i,j-1,k) ) &
       + ci*( cd2(i,j+3,k) + cd2(i,j-2,k) ) )

ux = gi*(ai*( cd3(i,j+1,k) + cd3(i,j,k) ) &
       + bi*( cd3(i,j+2,k) + cd3(i,j-1,k) ) &
       + ci*( cd3(i,j+3,k) + cd3(i,j-2,k) ) )

wz = gi*(ai*( cd4(i,j+1,k) + cd4(i,j,k) ) &
       + bi*( cd4(i,j+2,k) + cd4(i,j-1,k) ) &
       + ci*( cd4(i,j+3,k) + cd4(i,j-2,k) ) )
                     
         
ty = g*(a*dy1*( t(i,j+1,k) - t(i,j,k) ) &
      + b*dy3*( t(i,j+2,k) - t(i,j-1,k) ) &
      + c*dy5*( t(i,j+3,k) - t(i,j-2,k) ) )


u  = gi*(ai*( q(i,j+1,k,2)/q(i,j+1,k,1) + q(i,j,k,2)/q(i,j,k,1) ) &
       + bi*( q(i,j+2,k,2)/q(i,j+2,k,1) + q(i,j-1,k,2)/q(i,j-1,k,1) ) &
       + ci*( q(i,j+3,k,2)/q(i,j+3,k,1) + q(i,j-2,k,2)/q(i,j-2,k,1) ) )
      
v  = gi*(ai*( q(i,j+1,k,3)/q(i,j+1,k,1) + q(i,j,k,3)/q(i,j,k,1) ) &
       + bi*( q(i,j+2,k,3)/q(i,j+2,k,1) + q(i,j-1,k,3)/q(i,j-1,k,1) ) &
       + ci*( q(i,j+3,k,3)/q(i,j+3,k,1) + q(i,j-2,k,3)/q(i,j-2,k,1) ) )

w  = gi*(ai*( q(i,j+1,k,4)/q(i,j+1,k,1) + q(i,j,k,4)/q(i,j,k,1) ) &
       + bi*( q(i,j+2,k,4)/q(i,j+2,k,1) + q(i,j-1,k,4)/q(i,j-1,k,1) ) &
       + ci*( q(i,j+3,k,4)/q(i,j+3,k,1) + q(i,j-2,k,4)/q(i,j-2,k,1) ) )

muy = gi*(ai*( mu(i,j+1,k) + mu(i,j,k) ) &
        + bi*( mu(i,j+2,k) + mu(i,j-1,k) ) &
        + ci*( mu(i,j+3,k) + mu(i,j-2,k) ) )


txy = g1*muy*(uy + vx)
tyy = g2*muy*(2.0d0*vy - ux - wz) 
tyz = g1*muy*(vz + wy)
qy  = g3*muy*ty

vg(i,j,k,1) = 0.0d0
vg(i,j,k,2) = txy
vg(i,j,k,3) = tyy
vg(i,j,k,4) = tyz
vg(i,j,k,5) = u*txy + v*tyy + w*tyz - qy 


end do
end do
end do

! compute RHS contribution due to viscous term (central difference & finite volume)
do m=1,5
do k=1,nz  
do j=1,ny
do i=1,nx

s(i,j,k,m) = s(i,j,k,m) + (vg(i,j,k,m)-vg(i,j-1,k,m))/dy 
                        
end do
end do
end do
end do

deallocate(cd1,cd2,cd3,cd4)

!-------------------------!
!In z-direction:
!-------------------------!
allocate(cd1(1:nx,1:ny,-2:nz+3),cd2(1:nx,1:ny,-2:nz+3))
allocate(cd3(1:nx,1:ny,-2:nz+3),cd4(1:nx,1:ny,-2:nz+3))
!compute cross-derivatives: wx,wy,ux,vy
do k=-2,nz+3
do j=1,ny
do i=1,nx

!wx
cd1(i,j,k) = gd*(ad*dx2*( q(i+1,j,k,4)/q(i+1,j,k,1) - q(i-1,j,k,4)/q(i-1,j,k,1) ) &
               + bd*dx4*( q(i+2,j,k,4)/q(i+2,j,k,1) - q(i-2,j,k,4)/q(i-2,j,k,1) ) &
               + cd*dx6*( q(i+3,j,k,4)/q(i+3,j,k,1) - q(i-3,j,k,4)/q(i-3,j,k,1) ) )

!wy
cd2(i,j,k) = gd*(ad*dy2*( q(i,j+1,k,4)/q(i,j+1,k,1) - q(i,j-1,k,4)/q(i,j-1,k,1) ) &
               + bd*dy4*( q(i,j+2,k,4)/q(i,j+2,k,1) - q(i,j-2,k,4)/q(i,j-2,k,1) ) &
               + cd*dy6*( q(i,j+3,k,4)/q(i,j+3,k,1) - q(i,j-3,k,4)/q(i,j-3,k,1) ) )


!ux
cd3(i,j,k) = gd*(ad*dx2*( q(i+1,j,k,2)/q(i+1,j,k,1) - q(i-1,j,k,2)/q(i-1,j,k,1) ) &
               + bd*dx4*( q(i+2,j,k,2)/q(i+2,j,k,1) - q(i-2,j,k,2)/q(i-2,j,k,1) ) &
               + cd*dx6*( q(i+3,j,k,2)/q(i+3,j,k,1) - q(i-3,j,k,2)/q(i-3,j,k,1) ) )

!vy      
cd4(i,j,k) = gd*(ad*dy2*( q(i,j+1,k,3)/q(i,j+1,k,1) - q(i,j-1,k,3)/q(i,j-1,k,1) ) &
               + bd*dy4*( q(i,j+2,k,3)/q(i,j+2,k,1) - q(i,j-2,k,3)/q(i,j-2,k,1) ) &
               + cd*dy6*( q(i,j+3,k,3)/q(i,j+3,k,1) - q(i,j-3,k,3)/q(i,j-3,k,1) ) )
      
end do
end do
end do
      
!compute viscous fluxes at cell interfaces:

do k=0,nz
do j=1,ny
do i=1,nx

uz = g*(a*dz1*( q(i,j,k+1,2)/q(i,j,k+1,1) - q(i,j,k,2)/q(i,j,k,1) ) &
      + b*dz3*( q(i,j,k+2,2)/q(i,j,k+2,1) - q(i,j,k-1,2)/q(i,j,k-1,1) ) &
      + c*dz5*( q(i,j,k+3,2)/q(i,j,k+3,1) - q(i,j,k-2,2)/q(i,j,k-2,1) ) )
      
vz = g*(a*dz1*( q(i,j,k+1,3)/q(i,j,k+1,1) - q(i,j,k,3)/q(i,j,k,1) ) &
      + b*dz3*( q(i,j,k+2,3)/q(i,j,k+2,1) - q(i,j,k-1,3)/q(i,j,k-1,1) ) &
      + c*dz5*( q(i,j,k+3,3)/q(i,j,k+3,1) - q(i,j,k-2,3)/q(i,j,k-2,1) ) )
      
wz = g*(a*dz1*( q(i,j,k+1,4)/q(i,j,k+1,1) - q(i,j,k,4)/q(i,j,k,1) ) &
      + b*dz3*( q(i,j,k+2,4)/q(i,j,k+2,1) - q(i,j,k-1,4)/q(i,j,k-1,1) ) &
      + c*dz5*( q(i,j,k+3,4)/q(i,j,k+3,1) - q(i,j,k-2,4)/q(i,j,k-2,1) ) )
      

wx = gi*(ai*( cd1(i,j,k+1) + cd1(i,j,k) ) &
       + bi*( cd1(i,j,k+2) + cd1(i,j,k-1) ) &
       + ci*( cd1(i,j,k+3) + cd1(i,j,k-2) ) )

wy = gi*(ai*( cd2(i,j,k+1) + cd2(i,j,k) ) &
       + bi*( cd2(i,j,k+2) + cd2(i,j,k-1) ) &
       + ci*( cd2(i,j,k+3) + cd2(i,j,k-2) ) )

ux = gi*(ai*( cd3(i,j,k+1) + cd3(i,j,k) ) &
       + bi*( cd3(i,j,k+2) + cd3(i,j,k-1) ) &
       + ci*( cd3(i,j,k+3) + cd3(i,j,k-2) ) )

vy = gi*(ai*( cd4(i,j,k+1) + cd4(i,j,k) ) &
       + bi*( cd4(i,j,k+2) + cd4(i,j,k-1) ) &
       + ci*( cd4(i,j,k+3) + cd4(i,j,k-2) ) )
                     
         
tz = g*(a*dz1*( t(i,j,k+1) - t(i,j,k) ) &
      + b*dz3*( t(i,j,k+2) - t(i,j,k-1) ) &
      + c*dz5*( t(i,j,k+3) - t(i,j,k-2) ) )   


u  = gi*(ai*( q(i,j,k+1,2)/q(i,j,k+1,1) + q(i,j,k,2)/q(i,j,k,1) ) &
       + bi*( q(i,j,k+2,2)/q(i,j,k+2,1) + q(i,j,k-1,2)/q(i,j,k-1,1) ) &
       + ci*( q(i,j,k+3,2)/q(i,j,k+3,1) + q(i,j,k-2,2)/q(i,j,k-2,1) ) )
      
v  = gi*(ai*( q(i,j,k+1,3)/q(i,j,k+1,1) + q(i,j,k,3)/q(i,j,k,1) ) &
       + bi*( q(i,j,k+2,3)/q(i,j,k+2,1) + q(i,j,k-1,3)/q(i,j,k-1,1) ) &
       + ci*( q(i,j,k+3,3)/q(i,j,k+3,1) + q(i,j,k-2,3)/q(i,j,k-2,1) ) )

w  = gi*(ai*( q(i,j,k+1,4)/q(i,j,k+1,1) + q(i,j,k,4)/q(i,j,k,1) ) &
       + bi*( q(i,j,k+2,4)/q(i,j,k+2,1) + q(i,j,k-1,4)/q(i,j,k-1,1) ) &
       + ci*( q(i,j,k+3,4)/q(i,j,k+3,1) + q(i,j,k-2,4)/q(i,j,k-2,1) ) )

muz = gi*(ai*( mu(i,j,k+1) + mu(i,j,k) ) &
        + bi*( mu(i,j,k+2) + mu(i,j,k-1) ) &
        + ci*( mu(i,j,k+3) + mu(i,j,k-2) ) )


txz = g1*muz*(uz + wx)
tyz = g1*muz*(vz + wy) 
tzz = g2*muz*(2.0d0*wz - ux - vy)
qz  = g3*muz*tz

vh(i,j,k,1) = 0.0d0
vh(i,j,k,2) = txz
vh(i,j,k,3) = tyz
vh(i,j,k,4) = tzz
vh(i,j,k,5) = u*txz + v*tyz + w*tzz - qz


end do
end do
end do

! compute RHS contribution due to viscous term (central difference & finite volume)
do m=1,5
do k=1,nz  
do j=1,ny
do i=1,nx

s(i,j,k,m) = s(i,j,k,m) + (vh(i,j,k,m)-vh(i,j,k-1,m))/dz 
                        
end do
end do
end do
end do

deallocate(t,mu,vf,vg,vh)
deallocate(cd1,cd2,cd3,cd4)
return
end



!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS) for viscous term (central) finite volume
!not recommended
!-----------------------------------------------------------------------------------!
subroutine rhsVIS_old(nx,ny,nz,dx,dy,dz,q,s)
implicit none
integer::nx,ny,nz,i,j,k,m
real*8 ::dx,dy,dz,re,pr,gamma,Tref,ma
real*8 ::g,a,b,c,dx1,dx3,dx5,dy1,dy3,dy5,dz1,dz3,dz5
real*8 ::g1,g2,g3,g4,cc,u,v,w,txx,tyy,tzz,txy,txz,tyz,tx,ty,tz
real*8 ::ux,uy,uz,vx,vy,vz,wx,wy,wz,qx,qy,qz
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5),s(nx,ny,nz,5)
real*8,allocatable:: t(:,:,:),vf(:,:,:,:),vg(:,:,:,:),vh(:,:,:,:),mu(:,:,:)

common /viscosity/ re,pr,Tref
common /fluids/ gamma
common /mach/ ma

allocate(t(-2:nx+3,-2:ny+3,-2:nz+3),mu(-2:nx+3,-2:ny+3,-2:nz+3))
allocate(vf(0:nx,0:ny,0:nz,5),vg(0:nx,0:ny,0:nz,5),vh(0:nx,0:ny,0:nz,5))

!for derivatives (finite difference)
!g = 1.0d0/128.0d0
!a = 150.0d0
!b =-25.0d0
!c = 3.0d0

!for derivatives (finite volume)
g = 1.0d0/180.0d0
a = 245.0d0
b =-75.0d0
c = 10.0d0


g1 = 1.0d0/re
g2 = (2.0d0/3.0d0)/re
g3 =-1.0d0/(re*pr*ma*ma*(gamma-1.0d0))
g4 = (gamma-1.0d0)*gamma*ma*ma

dx1 = 1.0d0/dx
dx3 = 1.0d0/(3.0d0*dx)
dx5 = 1.0d0/(5.0d0*dx)

dy1 = 1.0d0/dy
dy3 = 1.0d0/(3.0d0*dy)
dy5 = 1.0d0/(5.0d0*dy)

dz1 = 1.0d0/dz
dz3 = 1.0d0/(3.0d0*dz)
dz5 = 1.0d0/(5.0d0*dz)

cc = 110.4d0/Tref

!compute temperature
do k=-2,nz+3
do j=-2,ny+3
do i=-2,nx+3
t(i,j,k) = g4*(q(i,j,k,5)/q(i,j,k,1) &
       - 0.5d0*((q(i,j,k,2)/q(i,j,k,1))*(q(i,j,k,2)/q(i,j,k,1)) &
               +(q(i,j,k,3)/q(i,j,k,1))*(q(i,j,k,3)/q(i,j,k,1)) &
               +(q(i,j,k,4)/q(i,j,k,1))*(q(i,j,k,4)/q(i,j,k,1)) ) )
               
mu(i,j,k) = (t(i,j,k)**(1.5))*(1.0d0+cc)/(t(i,j,k)+cc) !Sutherland's Law
               
end do
end do
end do

!compute viscous fluxes at cell interfaces:

do k=0,nz
do j=0,ny
do i=0,nx

ux = g*(  a*dx1*( q(i+1,j,k,2)/q(i+1,j,k,1) - q(i,j,k,2)/q(i,j,k,1) ) &
        + b*dx3*( q(i+2,j,k,2)/q(i+2,j,k,1) - q(i-1,j,k,2)/q(i-1,j,k,1) ) &
        + c*dx5*( q(i+3,j,k,2)/q(i+3,j,k,1) - q(i-2,j,k,2)/q(i-2,j,k,1) ) )

uy = g*(a*dy1*( q(i,j+1,k,2)/q(i,j+1,k,1) - q(i,j,k,2)/q(i,j,k,1) ) &
      + b*dy3*( q(i,j+2,k,2)/q(i,j+2,k,1) - q(i,j-1,k,2)/q(i,j-1,k,1) ) &
      + c*dy5*( q(i,j+3,k,2)/q(i,j+3,k,1) - q(i,j-2,k,2)/q(i,j-2,k,1) ) )


uz = g*(a*dz1*( q(i,j,k+1,2)/q(i,j,k+1,1) - q(i,j,k,2)/q(i,j,k,1) ) &
      + b*dz3*( q(i,j,k+2,2)/q(i,j,k+2,1) - q(i,j,k-1,2)/q(i,j,k-1,1) ) &
      + c*dz5*( q(i,j,k+3,2)/q(i,j,k+3,1) - q(i,j,k-2,2)/q(i,j,k-2,1) ) )


vx = g*(a*dx1*( q(i+1,j,k,3)/q(i+1,j,k,1) - q(i,j,k,3)/q(i,j,k,1) ) &
      + b*dx3*( q(i+2,j,k,3)/q(i+2,j,k,1) - q(i-1,j,k,3)/q(i-1,j,k,1) ) &
      + c*dx5*( q(i+3,j,k,3)/q(i+3,j,k,1) - q(i-2,j,k,3)/q(i-2,j,k,1) ) )

vy = g*(a*dy1*( q(i,j+1,k,3)/q(i,j+1,k,1) - q(i,j,k,3)/q(i,j,k,1) ) &
      + b*dy3*( q(i,j+2,k,3)/q(i,j+2,k,1) - q(i,j-1,k,3)/q(i,j-1,k,1) ) &
      + c*dy5*( q(i,j+3,k,3)/q(i,j+3,k,1) - q(i,j-2,k,3)/q(i,j-2,k,1) ) )
      
vz = g*(a*dz1*( q(i,j,k+1,3)/q(i,j,k+1,1) - q(i,j,k,3)/q(i,j,k,1) ) &
      + b*dz3*( q(i,j,k+2,3)/q(i,j,k+2,1) - q(i,j,k-1,3)/q(i,j,k-1,1) ) &
      + c*dz5*( q(i,j,k+3,3)/q(i,j,k+3,1) - q(i,j,k-2,3)/q(i,j,k-2,1) ) )

wx = g*(a*dx1*( q(i+1,j,k,4)/q(i+1,j,k,1) - q(i,j,k,4)/q(i,j,k,1) ) &
      + b*dx3*( q(i+2,j,k,4)/q(i+2,j,k,1) - q(i-1,j,k,4)/q(i-1,j,k,1) ) &
      + c*dx5*( q(i+3,j,k,4)/q(i+3,j,k,1) - q(i-2,j,k,4)/q(i-2,j,k,1) ) )

wy = g*(a*dy1*( q(i,j+1,k,4)/q(i,j+1,k,1) - q(i,j,k,4)/q(i,j,k,1) ) &
      + b*dy3*( q(i,j+2,k,4)/q(i,j+2,k,1) - q(i,j-1,k,4)/q(i,j-1,k,1) ) &
      + c*dy5*( q(i,j+3,k,4)/q(i,j+3,k,1) - q(i,j-2,k,4)/q(i,j-2,k,1) ) )
      
wz = g*(a*dz1*( q(i,j,k+1,4)/q(i,j,k+1,1) - q(i,j,k,4)/q(i,j,k,1) ) &
      + b*dz3*( q(i,j,k+2,4)/q(i,j,k+2,1) - q(i,j,k-1,4)/q(i,j,k-1,1) ) &
      + c*dz5*( q(i,j,k+3,4)/q(i,j,k+3,1) - q(i,j,k-2,4)/q(i,j,k-2,1) ) )



tx = g*(a*dx1*( t(i+1,j,k) - t(i,j,k) ) &
      + b*dx3*( t(i+2,j,k) - t(i-1,j,k) ) &
      + c*dx5*( t(i+3,j,k) - t(i-2,j,k) ) )

ty = g*(a*dy1*( t(i,j+1,k) - t(i,j,k) ) &
      + b*dy3*( t(i,j+2,k) - t(i,j-1,k) ) &
      + c*dy5*( t(i,j+3,k) - t(i,j-2,k) ) )

tz = g*(a*dz1*( t(i,j,k+1) - t(i,j,k) ) &
      + b*dz3*( t(i,j,k+2) - t(i,j,k-1) ) &
      + c*dz5*( t(i,j,k+3) - t(i,j,k-2) ) )      

txx = g2*(2.0d0*ux - vy - wz)*mu(i,j,k) 
tyy = g2*(2.0d0*vy - ux - wz)*mu(i,j,k) 
tzz = g2*(2.0d0*wz - ux - vy)*mu(i,j,k)
txy = g1*(uy + vx)*mu(i,j,k)
txz = g1*(uz + wx)*mu(i,j,k) 
tyz = g1*(vz + wy)*mu(i,j,k)

qx  = g3*tx*mu(i,j,k)
qy  = g3*ty*mu(i,j,k)         
qz  = g3*tz*mu(i,j,k)

u = q(i,j,k,2)/q(i,j,k,1)
v = q(i,j,k,3)/q(i,j,k,1)
w = q(i,j,k,4)/q(i,j,k,1)


vf(i,j,k,1) = 0.0d0
vf(i,j,k,2) = txx
vf(i,j,k,3) = txy
vf(i,j,k,4) = txz
vf(i,j,k,5) = u*txx + v*txy + w*txz - qx

vg(i,j,k,1) = 0.0d0
vg(i,j,k,2) = txy
vg(i,j,k,3) = tyy
vg(i,j,k,4) = tyz
vg(i,j,k,5) = u*txy + v*tyy + w*tyz - qy 

vh(i,j,k,1) = 0.0d0
vh(i,j,k,2) = txz
vh(i,j,k,3) = tyz
vh(i,j,k,4) = tzz
vh(i,j,k,5) = u*txz + v*tyz + w*tzz - qz 

end do
end do
end do

! compute RHS contribution due to viscous term (central difference & finite volume)
do m=1,5
do k=1,nz  
do j=1,ny
do i=1,nx

s(i,j,k,m) = s(i,j,k,m) + (vf(i,j,k,m)-vf(i-1,j,k,m))/dx &
                        + (vg(i,j,k,m)-vg(i,j-1,k,m))/dy &
                        + (vh(i,j,k,m)-vh(i,j,k-1,m))/dz

end do
end do
end do
end do

deallocate(t,mu,vf,vg,vh)

return
end


!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS) for inviscid term (central)
!-----------------------------------------------------------------------------------!
subroutine rhsCS(nx,ny,nz,dx,dy,dz,q,s)
implicit none
integer::nx,ny,nz,i,j,k,m,iprob
real*8 ::dx,dy,dz,gamma
real*8 ::g,a,b,c,gm,uu,vv,ww,pp,rh,re
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5),s(nx,ny,nz,5)
real*8,allocatable:: u(:,:,:),v(:,:,:),p(:,:,:),vf(:,:,:,:),vg(:,:,:,:),w(:,:,:)
real*8,allocatable::vh(:,:,:,:)

common /fluids/ gamma
common /problem/ iprob

allocate(u(-2:nx+3,-2:ny+3,-2:nz+3),v(-2:nx+3,-2:ny+3,-2:nz+3),p(-2:nx+3,-2:ny+3,-2:nz+3))
allocate(w(-2:nx+3,-2:ny+3,-2:nz+3))
allocate(vf(0:nx,0:ny,0:nz,5),vg(0:nx,0:ny,0:nz,5),vh(0:nx,0:ny,0:nz,5))

!for interpolation (finite difference)
!g = 1.0d0/256.0d0
!a = 150.0d0
!b =-25.0d0
!c = 3.0d0

!for interpolation (finite volume)
g = 1.0d0/60.0d0
a = 37.0d0
b =-8.0d0
c = 1.0d0

gm= gamma-1.0d0


!compute temperature
do k=-2,nz+3
do j=-2,ny+3
do i=-2,nx+3
u(i,j,k) = q(i,j,k,2)/q(i,j,k,1)
v(i,j,k) = q(i,j,k,3)/q(i,j,k,1)
w(i,j,k) = q(i,j,k,4)/q(i,j,k,1)
p(i,j,k) = gm*(q(i,j,k,5)- 0.5d0*(q(i,j,k,2)*u(i,j,k)+q(i,j,k,3)*v(i,j,k)+q(i,j,k,4)*w(i,j,k)))
end do
end do
end do

!compute fluxes at cell interfaces:x

do k=0,nz
do j=0,ny
do i=0,nx

uu = g*(a*( u(i+1,j,k) + u(i,j,k) ) &
      + b*( u(i+2,j,k) + u(i-1,j,k) ) &
      + c*( u(i+3,j,k) + u(i-2,j,k) ) )
      
vv = g*(a*( v(i+1,j,k) + v(i,j,k) ) &
      + b*( v(i+2,j,k) + v(i-1,j,k) ) &
      + c*( v(i+3,j,k) + v(i-2,j,k) ) )

ww = g*(a*( w(i+1,j,k) + w(i,j,k) ) &
      + b*( w(i+2,j,k) + w(i-1,j,k) ) &
      + c*( w(i+3,j,k) + w(i-2,j,k) ) )

      
pp = g*(a*( p(i+1,j,k) + p(i,j,k) ) &
      + b*( p(i+2,j,k) + p(i-1,j,k) ) &
      + c*( p(i+3,j,k) + p(i-2,j,k) ) )

rh = g*(a*( q(i+1,j,k,1) + q(i,j,k,1) ) &
      + b*( q(i+2,j,k,1) + q(i-1,j,k,1) ) &
      + c*( q(i+3,j,k,1) + q(i-2,j,k,1) ) )

re = g*(a*( q(i+1,j,k,5) + q(i,j,k,5) ) &
      + b*( q(i+2,j,k,5) + q(i-1,j,k,5) ) &
      + c*( q(i+3,j,k,5) + q(i-2,j,k,5) ) )

      
vf(i,j,k,1) = rh*uu
vf(i,j,k,2) = rh*uu*uu + pp
vf(i,j,k,3) = rh*uu*vv
vf(i,j,k,4) = rh*uu*ww
vf(i,j,k,5) = (re+pp)*uu

end do
end do
end do

!compute fluxes at cell interfaces:y

do k=0,nz
do j=0,ny
do i=0,nx

uu = g*(a*( u(i,j+1,k) + u(i,j,k) ) &
      + b*( u(i,j+2,k) + u(i,j-1,k) ) &
      + c*( u(i,j+3,k) + u(i,j-2,k) ) )
      
vv = g*(a*( v(i,j+1,k) + v(i,j,k) ) &
      + b*( v(i,j+2,k) + v(i,j-1,k) ) &
      + c*( v(i,j+3,k) + v(i,j-2,k) ) )

ww = g*(a*( w(i,j+1,k) + w(i,j,k) ) &
      + b*( w(i,j+2,k) + w(i,j-1,k) ) &
      + c*( w(i,j+3,k) + w(i,j-2,k) ) )
      
      
pp = g*(a*( p(i,j+1,k) + p(i,j,k) ) &
      + b*( p(i,j+2,k) + p(i,j-1,k) ) &
      + c*( p(i,j+3,k) + p(i,j-2,k) ) )

rh = g*(a*( q(i,j+1,k,1) + q(i,j,k,1) ) &
      + b*( q(i,j+2,k,1) + q(i,j-1,k,1) ) &
      + c*( q(i,j+3,k,1) + q(i,j-2,k,1) ) )

re = g*(a*( q(i,j+1,k,5) + q(i,j,k,5) ) &
      + b*( q(i,j+2,k,5) + q(i,j-1,k,5) ) &
      + c*( q(i,j+3,k,5) + q(i,j-2,k,5) ) )

      
vg(i,j,k,1) = rh*vv
vg(i,j,k,2) = rh*uu*vv 
vg(i,j,k,3) = rh*vv*vv + pp
vg(i,j,k,4) = rh*vv*ww
vg(i,j,k,5) = (re+pp)*vv

end do
end do
end do


!compute fluxes at cell interfaces:z

do k=0,nz
do j=0,ny
do i=0,nx

uu = g*(a*( u(i,j,k+1) + u(i,j,k) ) &
      + b*( u(i,j,k+2) + u(i,j,k-1) ) &
      + c*( u(i,j,k+3) + u(i,j,k-2) ) )
      
vv = g*(a*( v(i,j,k+1) + v(i,j,k) ) &
      + b*( v(i,j,k+2) + v(i,j,k-1) ) &
      + c*( v(i,j,k+3) + v(i,j,k-2) ) )

ww = g*(a*( w(i,j,k+1) + w(i,j,k) ) &
      + b*( w(i,j,k+2) + w(i,j,k-1) ) &
      + c*( w(i,j,k+3) + w(i,j,k-2) ) )
      
      
pp = g*(a*( p(i,j,k+1) + p(i,j,k) ) &
      + b*( p(i,j,k+2) + p(i,j,k-1) ) &
      + c*( p(i,j,k+3) + p(i,j,k-2) ) )

rh = g*(a*( q(i,j,k+1,1) + q(i,j,k,1) ) &
      + b*( q(i,j,k+2,1) + q(i,j,k-1,1) ) &
      + c*( q(i,j,k+3,1) + q(i,j,k-2,1) ) )

re = g*(a*( q(i,j,k+1,5) + q(i,j,k,5) ) &
      + b*( q(i,j,k+2,5) + q(i,j,k-1,5) ) &
      + c*( q(i,j,k+3,5) + q(i,j,k-2,5) ) )

      
vh(i,j,k,1) = rh*ww
vh(i,j,k,2) = rh*uu*ww 
vh(i,j,k,3) = rh*vv*ww 
vh(i,j,k,4) = rh*ww*ww + pp
vh(i,j,k,5) = (re+pp)*ww

end do
end do
end do



! compute RHS contribution due to inviscid term (central difference)
do m=1,5
do k=1,nz  
do j=1,ny
do i=1,nx

s(i,j,k,m) = - (vf(i,j,k,m)-vf(i-1,j,k,m))/dx &
             - (vg(i,j,k,m)-vg(i,j-1,k,m))/dy &
             - (vh(i,j,k,m)-vh(i,j,k-1,m))/dz

end do
end do
end do
end do

deallocate(u,v,p,vf,vg,vh)



return
end


!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS) for ILES (only inviscid term)
!-----------------------------------------------------------------------------------!
subroutine rhsILES(nx,ny,nz,dx,dy,dz,q,s)
implicit none
integer::nx,ny,nz,i,j,k,m,ix,iweno
real*8 ::dx,dy,dz
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5),s(nx,ny,nz,5)
real*8,allocatable:: u(:,:),uL(:,:),uR(:,:),hL(:,:),hR(:,:),rf(:,:)

common /fluxopt/ ix
common /weno_opt/ iweno

!---------------------------------!
!Compute x-fluxes for all j and k
!---------------------------------!
allocate(u(-2:nx+3,5))
allocate(uL(0:nx,5),uR(0:nx,5),hL(0:nx,5),hR(0:nx,5),rf(0:nx,5))

do k=1,nz
do j=1,ny
	
	!assign q vector as u in x-direction
    do m=1,5
    do i=-2,nx+3
    u(i,m)=q(i,j,k,m)
    end do
    end do
    
	!-----------------------------!
	!Reconstruction scheme:z
    !-----------------------------!
    !WENO5 construction
    call weno5r(nx,u,uL,uR)
    	
	!compute left and right fluxes
  	call xflux(nx,uL,hL)
	call xflux(nx,uR,hR)  
    
    !-----------------------------!
	!Riemann Solver: 
    !Rusanov flux in x-direction
    !-----------------------------!
    if (ix.eq.1) then
    call rusanov_x(nx,u,uL,uR,hL,hR,rf)
    else if (ix.eq.2) then
    call roe_x(nx,uL,uR,hL,hR,rf)
    else if (ix.eq.3) then
    call hll_x(nx,uL,uR,hL,hR,rf)
    else 
    call ausm_x(nx,uL,uR,rf)
    end if

    !-----------------------------!
	!Compute RHS contribution
    !-----------------------------!
	do m=1,5
	do i=1,nx 
	s(i,j,k,m)=-(rf(i,m)-rf(i-1,m))/dx     
	end do
    end do
    
end do
end do

deallocate(u,uL,uR,hL,hR,rf)

!---------------------------------!
!Compute y-fluxes for all i and k
!---------------------------------!
allocate(u(-2:ny+3,5))
allocate(uL(0:ny,5),uR(0:ny,5),hL(0:ny,5),hR(0:ny,5),rf(0:ny,5))


do i=1,ny
do k=1,nz
	
	!assign q vector as u in y-direction
    do m=1,5
    do j=-2,ny+3
    u(j,m)=q(i,j,k,m)
    end do
    end do

    !-----------------------------!
	!Reconstruction scheme:z
    !-----------------------------!
    !WENO5 construction
    call weno5r(ny,u,uL,uR)
    	
	
	!compute left and right fluxes
  	call yflux(ny,uL,hL)
	call yflux(ny,uR,hR)  


	!-----------------------------!
    !Riemann Solver: 
	!Rusanov flux in y-direction
    !-----------------------------!   
    if (ix.eq.1) then
    call rusanov_y(ny,u,uL,uR,hL,hR,rf)
    else if (ix.eq.2) then
    call roe_y(ny,uL,uR,hL,hR,rf)
    else if (ix.eq.3) then
    call hll_y(ny,uL,uR,hL,hR,rf)
    else
    call ausm_y(ny,uL,uR,rf)
    end if
    
	!-----------------------------!
	!Compute RHS contribution
    !-----------------------------!
	do m=1,5
	do j=1,ny 
	s(i,j,k,m)=s(i,j,k,m)-(rf(j,m)-rf(j-1,m))/dy     
	end do
    end do
       
end do
end do


deallocate(u,uL,uR,hL,hR,rf)

!---------------------------------!
!Compute z-fluxes for all j and k
!---------------------------------!
allocate(u(-2:nz+3,5))
allocate(uL(0:nz,5),uR(0:nz,5),hL(0:nz,5),hR(0:nz,5),rf(0:nz,5))


do j=1,ny
do i=1,nx
	
	!assign q vector as u in z-direction
    do m=1,5
    do k=-2,nz+3
    u(k,m)=q(i,j,k,m)
    end do
    end do

    !-----------------------------!
	!Reconstruction scheme:z
    !-----------------------------!
    !WENO5 construction
    call weno5r(nz,u,uL,uR)
    

	!compute left and right fluxes
  	call zflux(nz,uL,hL)
	call zflux(nz,uR,hR)  


	!-----------------------------!
    !Riemann Solver: 
	!Rusanov flux in z-direction
    !-----------------------------!   
    if (ix.eq.1) then
    call rusanov_z(nz,u,uL,uR,hL,hR,rf)
    else if (ix.eq.2) then
    call roe_z(nz,uL,uR,hL,hR,rf)
    else if (ix.eq.3) then
    call hll_z(nz,uL,uR,hL,hR,rf)
    else
    call ausm_z(nz,uL,uR,rf) 
    end if
    
	!-----------------------------!
	!Compute RHS contribution
    !-----------------------------!
	do m=1,5
	do k=1,nz 
	s(i,j,k,m)=s(i,j,k,m)-(rf(k,m)-rf(k-1,m))/dz    
	end do
    end do
       
end do
end do


deallocate(u,uL,uR,hL,hR,rf)

return
end



!-----------------------------------------------------------------------------------!
!WENO reconstruction: 5th order
!u:  cell centered data
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!-----------------------------------------------------------------------------------!
subroutine weno5r(n,u,uL,uR)
implicit none
integer::n,iweno
real*8 ::u(-2:n+3,5),uL(0:n,5),uR(0:n,5)

common /weno_opt/ iweno


	if (iweno.eq.0) then
        call linearR(n,u,uL,uR) !Linear schemes
    else if (iweno.eq.1) then
		call jweno5(n,u,uL,uR) 	!WENO-JS
    else if (iweno.eq.2) then
    	call zweno5(n,u,uL,uR) 	!WENO-Z
    else if (iweno.eq.3) then  	
    	call bweno5(n,u,uL,uR)  !Bandwidth optimized
    else if (iweno.eq.4) then
    	call cweno5(n,u,uL,uR)	!Central WENO (6th)
    else if (iweno.eq.5) then
    	call dweno5(n,u,uL,uR)	!DRP WWENO
    else
		call weno5(n,u,uL,uR)	!old coding for JS scheme
    end if


return
end


!-----------------------------------------------------------------------------------!
!Linear reconstructions:
!u:  cell centered data
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!-----------------------------------------------------------------------------------!
subroutine linearR(n,u,uL,uR)
implicit none
integer::n,iupw
real*8 ::u(-2:n+3,5),uL(0:n,5),uR(0:n,5)

common /upw_opt/ iupw

	if (iupw.eq.0) then
        call central6(n,u,uL,uR)	!Central 6th-order
    else if (iupw.eq.1) then
		call upwind5(n,u,uL,uR) 	!Fifth order upwind
    else if (iupw.eq.2) then
    	call drp4(n,u,uL,uR) 		!DRP
    else if (iupw.eq.3) then
    	call bandwidth4(n,u,uL,uR)  !Bandwidth optimized
    else
		call upwind5(n,u,uL,uR) 	!Fifth order upwind
    end if



return
end



!-----------------------------------------------------------------------------------!
!Riemann Solver: Rusanov flux
!u:  cell centered data
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine rusanov_x(n,u,uL,uR,hL,hR,rf)
implicit none
integer::n,m,i
real*8 ::u(-2:n+3,5),uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5)
real*8 ::l1,l2,l3,rad0,rad1,ps,p,a
real*8 ::gamma
    
common /fluids/ gamma

    
do i=0,n
  	
	!at point i
    p = (gamma-1.0d0)*(u(i,5)-0.5d0*(u(i,2)*u(i,2)/u(i,1) &
                                    +u(i,3)*u(i,3)/u(i,1) &
                                    +u(i,4)*u(i,4)/u(i,1) ))
	a = dsqrt(gamma*p/u(i,1)) 
    
	l1=dabs(u(i,2)/u(i,1))
	l2=dabs(u(i,2)/u(i,1) + a)
	l3=dabs(u(i,2)/u(i,1) - a)
	rad0 = max(l1,l2,l3)

    !at point i+1
    p = (gamma-1.0d0)*(u(i+1,5)-0.5d0*(u(i+1,2)*u(i+1,2)/u(i+1,1) &
                                      +u(i+1,3)*u(i+1,3)/u(i+1,1) &
                                      +u(i+1,4)*u(i+1,4)/u(i+1,1) ))
	a = dsqrt(gamma*p/u(i+1,1)) 
    
	l1=dabs(u(i+1,2)/u(i+1,1))
	l2=dabs(u(i+1,2)/u(i+1,1) + a)
	l3=dabs(u(i+1,2)/u(i+1,1) - a)
	rad1 = max(l1,l2,l3)
    
		!characteristic speed for Rusanov flux
		ps = max(rad0,rad1)

		!compute flux 
		do m=1,5  
    	rf(i,m)=0.5d0*((hR(i,m)+hL(i,m)) - ps*(uR(i,m)-uL(i,m)))      
		end do
    
end do
    
return
end


!-----------------------------------------------------------------------------------!
!Riemann Solver: Rusanov flux
!u:  cell centered data
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine rusanov_y(n,u,uL,uR,hL,hR,rf)
implicit none
integer::n,m,i
real*8 ::u(-2:n+3,5),uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5)
real*8 ::l1,l2,l3,rad0,rad1,ps,p,a
real*8 ::gamma
    
common /fluids/ gamma

    
do i=0,n
  	
	!at point i
    p = (gamma-1.0d0)*(u(i,5)-0.5d0*(u(i,2)*u(i,2)/u(i,1) &
                                    +u(i,3)*u(i,3)/u(i,1) &
                                    +u(i,4)*u(i,4)/u(i,1) ))
	a = dsqrt(gamma*p/u(i,1)) 
    
	l1=dabs(u(i,3)/u(i,1))
	l2=dabs(u(i,3)/u(i,1) + a)
	l3=dabs(u(i,3)/u(i,1) - a)
	rad0 = max(l1,l2,l3)

    !at point i+1
    p = (gamma-1.0d0)*(u(i+1,5)-0.5d0*(u(i+1,2)*u(i+1,2)/u(i+1,1) &
                                      +u(i+1,3)*u(i+1,3)/u(i+1,1) &
                                      +u(i+1,4)*u(i+1,4)/u(i+1,1) ))
	a = dsqrt(gamma*p/u(i+1,1)) 
    
	l1=dabs(u(i+1,3)/u(i+1,1))
	l2=dabs(u(i+1,3)/u(i+1,1) + a)
	l3=dabs(u(i+1,3)/u(i+1,1) - a)
	rad1 = max(l1,l2,l3)
    
		!characteristic speed for Rusanov flux
		ps = max(rad0,rad1)

		!compute flux 
		do m=1,5  
    	rf(i,m)=0.5d0*((hR(i,m)+hL(i,m)) - ps*(uR(i,m)-uL(i,m)))      
		end do
    
end do
    
return
end

!-----------------------------------------------------------------------------------!
!Riemann Solver: Rusanov flux
!u:  cell centered data
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine rusanov_z(n,u,uL,uR,hL,hR,rf)
implicit none
integer::n,m,i
real*8 ::u(-2:n+3,5),uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5)
real*8 ::l1,l2,l3,rad0,rad1,ps,p,a
real*8 ::gamma
    
common /fluids/ gamma

    
do i=0,n
  	
	!at point i
    p = (gamma-1.0d0)*(u(i,5)-0.5d0*(u(i,2)*u(i,2)/u(i,1) &
                                    +u(i,3)*u(i,3)/u(i,1) &
                                    +u(i,4)*u(i,4)/u(i,1) ))
	a = dsqrt(gamma*p/u(i,1)) 
    
	l1=dabs(u(i,4)/u(i,1))
	l2=dabs(u(i,4)/u(i,1) + a)
	l3=dabs(u(i,4)/u(i,1) - a)
	rad0 = max(l1,l2,l3)

    !at point i+1
    p = (gamma-1.0d0)*(u(i+1,5)-0.5d0*(u(i+1,2)*u(i+1,2)/u(i+1,1) &
                                      +u(i+1,3)*u(i+1,3)/u(i+1,1) &
                                      +u(i+1,4)*u(i+1,4)/u(i+1,1) ))
	a = dsqrt(gamma*p/u(i+1,1)) 
    
	l1=dabs(u(i+1,4)/u(i+1,1))
	l2=dabs(u(i+1,4)/u(i+1,1) + a)
	l3=dabs(u(i+1,4)/u(i+1,1) - a)
	rad1 = max(l1,l2,l3)
    
		!characteristic speed for Rusanov flux
		ps = max(rad0,rad1)

		!compute flux 
		do m=1,5  
    	rf(i,m)=0.5d0*((hR(i,m)+hL(i,m)) - ps*(uR(i,m)-uL(i,m)))      
		end do
    
end do
    
return
end


!-----------------------------------------------------------------------------------!
!Riemann Solver: Roe flux
!u:  cell centered data
!uL: weno reconstructed values at left state (positive)
!uR: weno reconstructed values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine roe_x(n,uL,uR,hL,hR,rf)
implicit none

integer::n,m,i
real*8 ::uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5),cr(5)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,hhLL,rhRR,uuRR,vvRR,eeRR,ppRR,hhRR,wwRR
real*8 ::uu,vv,ww,hh,aa,g1,g2,g3,g4,g5,q1,q2,q3,q4,q5
real*8 ::a11,a12,a13,a14,a15,a21,a22,a23,a24,a25,a31,a32,a33,a34,a35,a41,a42,a43,a44,a45
real*8 ::r11,r12,r13,r14,r15,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45
real*8 ::l11,l12,l13,l14,l15,l21,l22,l23,l24,l25,l31,l32,l33,l34,l35,l41,l42,l43,l44,l45
real*8 ::a51,a52,a53,a54,a55,r51,r52,r53,r54,r55,l51,l52,l53,l54,l55
real*8 ::capVs

common /fluids/ gamma

	!-----------------------------!
	!Roe flux in x-direction
    !-----------------------------!
	do i=0,n

	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
   	wwLL = uL(i,4)/rhLL	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    hhLL = eeLL + ppLL/rhLL

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    hhRR = eeRR + ppRR/rhRR

	!Roe averages

	uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
    ww = (dsqrt(rhLL)*wwLL + dsqrt(rhRR)*wwRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv+ww*ww)))

    !eigenvalues
    g1 = dabs(uu - aa)
    g2 = dabs(uu)
    g3 = dabs(uu)
    g4 = dabs(uu)
    g5 = dabs(uu + aa)

	!right eigenvectors
    r11 = 1.0d0
    r12 = 1.0d0
    r13 = 0.0d0
    r14 = 0.0d0
    r15 = 1.0d0

    r21 = uu-aa
    r22 = uu
    r23 = 0.0d0
    r24 = 0.0d0
    r25 = uu+aa    
    
	r31 = vv
    r32 = vv
    r33 = 1.0d0
    r34 = 0.0d0
    r35 = vv

	r41 = ww
    r42 = ww
    r43 = 0.0d0
    r44 = 1.0d0
    r45 = ww

    r51 = hh - uu*aa
	r52 = 0.5d0*(uu*uu+vv*vv+ww*ww)
    r53 = vv
    r54 = ww
    r55 = hh + uu*aa
	
	capVs = uu*uu+vv*vv+ww*ww

	!left eigenvectors
    l11 = (hh*uu+0.5d0*(capVs)*(aa-uu))/(2.0d0*aa*hh-aa*(capVs))
    l12 = (-hh-aa*uu+0.5d0*(capVs))/(2.0d0*aa*hh-aa*capVs)
    l13 =-(aa*vv)/(2.0d0*aa*hh-aa*capVs)
    l14 =-(aa*ww)/(2.0d0*aa*hh-aa*capVs)
    l15 = (aa)/(2.0d0*aa*hh-aa*capVs)

    l21 = 2.0d0*aa*(hh-capVs)/(2.0d0*aa*hh-aa*capVs)
    l22 = 2.0d0*aa*uu/(2.0d0*aa*hh-aa*capVs)
    l23 = 2.0d0*aa*vv/(2.0d0*aa*hh-aa*capVs)
    l24 = 2.0d0*aa*ww/(2.0d0*aa*hh-aa*capVs)
    l25 =-2.0d0*aa/(2.0d0*aa*hh-aa*capVs)    
    
	l31 = aa*vv*(-2.0d0*hh+capVs)/(2.0d0*aa*hh-aa*capVs)
    l32 = 0.0d0
    l33 = 1.0d0
    l34 = 0.0d0
    l35 = 0.0d0

	l41 = aa*ww*(-2.0d0*hh+capVs)/(2.0d0*aa*hh-aa*capVs)
    l42 = 0.0d0
    l43 = 0.0d0
    l44 = 1.0d0
    l45 = 0.0d0

    l51 = (capVs*0.5d0*(uu+aa)-uu*hh)/(2.0d0*aa*hh-aa*(capVs))
	l52 = (hh-aa*uu-0.5d0*capVs)/(2.0d0*aa*hh-aa*(capVs))
    l53 = -aa*vv/(2.0d0*aa*hh-aa*(capVs))
    l54 = -aa*ww/(2.0d0*aa*hh-aa*(capVs))
    l55 = aa/(2.0d0*aa*hh-aa*(capVs))

    a11 = g1*l11*r11 + g2*l21*r12 + g3*l31*r13 + g4*l41*r14 + g5*l51*r15
    a12 = g1*l12*r11 + g2*l22*r12 + g3*l32*r13 + g4*l42*r14 + g5*l52*r15
    a13 = g1*l13*r11 + g2*l23*r12 + g3*l33*r13 + g4*l43*r14 + g5*l53*r15
    a14 = g1*l14*r11 + g2*l24*r12 + g3*l34*r13 + g4*l44*r14 + g5*l54*r15
    a15 = g1*l15*r11 + g2*l25*r12 + g3*l35*r13 + g4*l45*r14 + g5*l55*r15    

    a21 = g1*l11*r21 + g2*l21*r22 + g3*l31*r23 + g4*l41*r24 + g5*l51*r25
    a22 = g1*l12*r21 + g2*l22*r22 + g3*l32*r23 + g4*l42*r24 + g5*l52*r25
    a23 = g1*l13*r21 + g2*l23*r22 + g3*l33*r23 + g4*l43*r24 + g5*l53*r25
    a24 = g1*l14*r21 + g2*l24*r22 + g3*l34*r23 + g4*l44*r24 + g5*l54*r25
    a25 = g1*l15*r21 + g2*l25*r22 + g3*l35*r23 + g4*l45*r24 + g5*l55*r25   

    a31 = g1*l11*r31 + g2*l21*r32 + g3*l31*r33 + g4*l41*r34 + g5*l51*r35
    a32 = g1*l12*r31 + g2*l22*r32 + g3*l32*r33 + g4*l42*r34 + g5*l52*r35
    a33 = g1*l13*r31 + g2*l23*r32 + g3*l33*r33 + g4*l43*r34 + g5*l53*r35
    a34 = g1*l14*r31 + g2*l24*r32 + g3*l34*r33 + g4*l44*r34 + g5*l54*r35
    a35 = g1*l15*r31 + g2*l25*r32 + g3*l35*r33 + g4*l45*r34 + g5*l55*r35    

    a41 = g1*l11*r41 + g2*l21*r42 + g3*l31*r43 + g4*l41*r44 + g5*l51*r45
    a42 = g1*l12*r41 + g2*l22*r42 + g3*l32*r43 + g4*l42*r44 + g5*l52*r45
    a43 = g1*l13*r41 + g2*l23*r42 + g3*l33*r43 + g4*l43*r44 + g5*l53*r45
    a44 = g1*l14*r41 + g2*l24*r42 + g3*l34*r43 + g4*l44*r44 + g5*l54*r45
    a45 = g1*l15*r41 + g2*l25*r42 + g3*l35*r43 + g4*l45*r44 + g5*l55*r45    

    a51 = g1*l11*r51 + g2*l21*r52 + g3*l31*r53 + g4*l41*r54 + g5*l51*r55
    a52 = g1*l12*r51 + g2*l22*r52 + g3*l32*r53 + g4*l42*r54 + g5*l52*r55
    a53 = g1*l13*r51 + g2*l23*r52 + g3*l33*r53 + g4*l43*r54 + g5*l53*r55
    a54 = g1*l14*r51 + g2*l24*r52 + g3*l34*r53 + g4*l44*r54 + g5*l54*r55
    a55 = g1*l15*r51 + g2*l25*r52 + g3*l35*r53 + g4*l45*r54 + g5*l55*r55    

    
	q1 = uR(i,1)-uL(i,1)
    q2 = uR(i,2)-uL(i,2)
    q3 = uR(i,3)-uL(i,3)
    q4 = uR(i,4)-uL(i,4)
    q5 = uR(i,5)-uL(i,5)    
    
    cr(1) = q1*a11 + q2*a12 + q3*a13 + q4*a14 + q5*a15
    cr(2) = q1*a21 + q2*a22 + q3*a23 + q4*a24 + q5*a25
    cr(3) = q1*a31 + q2*a32 + q3*a33 + q4*a34 + q5*a35
    cr(4) = q1*a41 + q2*a42 + q3*a43 + q4*a44 + q5*a45
    cr(5) = q1*a51 + q2*a52 + q3*a53 + q4*a54 + q5*a55

    	!compute flux in x-direction
    	do m=1,5	
		rf(i,m)=0.5d0*(hR(i,m)+hL(i,m)) - 0.5d0*cr(m)     
    	end do
    
    
	end do

return
end


!-----------------------------------------------------------------------------------!
!Riemann Solver: Roe flux
!u:  cell centered data
!uL: weno reconstructed values at left state (positive)
!uR: weno reconstructed values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine roe_y(n,uL,uR,hL,hR,rf)
implicit none

integer::n,m,i
real*8 ::uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5),cr(5)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,hhLL,rhRR,uuRR,vvRR,eeRR,ppRR,hhRR,wwRR
real*8 ::uu,vv,ww,hh,aa,g1,g2,g3,g4,g5,q1,q2,q3,q4,q5
real*8 ::a11,a12,a13,a14,a15,a21,a22,a23,a24,a25,a31,a32,a33,a34,a35,a41,a42,a43,a44,a45
real*8 ::r11,r12,r13,r14,r15,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45
real*8 ::l11,l12,l13,l14,l15,l21,l22,l23,l24,l25,l31,l32,l33,l34,l35,l41,l42,l43,l44,l45
real*8 ::a51,a52,a53,a54,a55,r51,r52,r53,r54,r55,l51,l52,l53,l54,l55
real*8 ::capVs

common /fluids/ gamma

	!-----------------------------!
	!Roe flux in y-direction
    !-----------------------------!
	do i=0,n

	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
   	wwLL = uL(i,4)/rhLL	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    hhLL = eeLL + ppLL/rhLL

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    hhRR = eeRR + ppRR/rhRR

	!Roe averages

	uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
    ww = (dsqrt(rhLL)*wwLL + dsqrt(rhRR)*wwRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv+ww*ww)))

    !eigenvalues
    g1 = dabs(vv - aa)
    g2 = dabs(vv)
    g3 = dabs(vv)
    g4 = dabs(vv)
    g5 = dabs(vv + aa)

	!right eigenvectors
    r11 = 1.0d0
    r12 = 0.0d0
    r13 = 1.0d0
    r14 = 0.0d0
    r15 = 1.0d0

    r21 = uu
    r22 = 1.0d0
    r23 = uu
    r24 = 0.0d0
    r25 = uu    
    
	r31 = vv-aa
    r32 = 0.0d0
    r33 = vv
    r34 = 0.0d0
    r35 = vv+aa

	r41 = ww
    r42 = 0.0d0
    r43 = ww
    r44 = 1.0d0
    r45 = ww

    r51 = hh - vv*aa
	r52 = uu
    r53 = 0.5d0*(uu*uu+vv*vv+ww*ww)
    r54 = ww
    r55 = hh + vv*aa
	
	capVs = uu*uu+vv*vv+ww*ww

	!left eigenvectors
    l11 = (hh*vv+0.5d0*(capVs)*(aa-vv))/(2.0d0*aa*hh-aa*(capVs))
    l12 =-(aa*uu)/(2.0d0*aa*hh-aa*capVs)
    l13 = (-hh-aa*vv+0.5d0*(capVs))/(2.0d0*aa*hh-aa*capVs)
    l14 =-(aa*ww)/(2.0d0*aa*hh-aa*capVs)
    l15 = (aa)/(2.0d0*aa*hh-aa*capVs)

    l21 = (-2.0d0*hh+capVs)*(uu*aa)/(2.0d0*aa*hh-aa*capVs)
    l22 = 1.0d0
    l23 = 0.0d0
    l24 = 0.0d0
    l25 = 0.0d0
    
	l31 = 2.0d0*aa*(hh-capVs)/(2.0d0*aa*hh-aa*capVs)
    l32 = 2.0d0*aa*uu/(2.0d0*aa*hh-aa*capVs)
    l33 = 2.0d0*aa*vv/(2.0d0*aa*hh-aa*capVs)
    l34 = 2.0d0*aa*ww/(2.0d0*aa*hh-aa*capVs)
    l35 =-2.0d0*aa/(2.0d0*aa*hh-aa*capVs)

	l41 = aa*ww*(-2.0d0*hh+capVs)/(2.0d0*aa*hh-aa*capVs)
    l42 = 0.0d0
    l43 = 0.0d0
    l44 = 1.0d0
    l45 = 0.0d0

    l51 = (capVs*0.5d0*(vv+aa)-vv*hh)/(2.*aa*hh-aa*(capVs))
	l52 = -aa*uu/(2.0d0*aa*hh-aa*(capVs))
    l53 = (hh-aa*vv-0.5d0*capVs)/(2.0d0*aa*hh-aa*(capVs))
    l54 = -aa*ww/(2.0d0*aa*hh-aa*(capVs))
    l55 = aa/(2.0d0*aa*hh-aa*(capVs))

    a11 = g1*l11*r11 + g2*l21*r12 + g3*l31*r13 + g4*l41*r14 + g5*l51*r15
    a12 = g1*l12*r11 + g2*l22*r12 + g3*l32*r13 + g4*l42*r14 + g5*l52*r15
    a13 = g1*l13*r11 + g2*l23*r12 + g3*l33*r13 + g4*l43*r14 + g5*l53*r15
    a14 = g1*l14*r11 + g2*l24*r12 + g3*l34*r13 + g4*l44*r14 + g5*l54*r15
    a15 = g1*l15*r11 + g2*l25*r12 + g3*l35*r13 + g4*l45*r14 + g5*l55*r15    

    a21 = g1*l11*r21 + g2*l21*r22 + g3*l31*r23 + g4*l41*r24 + g5*l51*r25
    a22 = g1*l12*r21 + g2*l22*r22 + g3*l32*r23 + g4*l42*r24 + g5*l52*r25
    a23 = g1*l13*r21 + g2*l23*r22 + g3*l33*r23 + g4*l43*r24 + g5*l53*r25
    a24 = g1*l14*r21 + g2*l24*r22 + g3*l34*r23 + g4*l44*r24 + g5*l54*r25
    a25 = g1*l15*r21 + g2*l25*r22 + g3*l35*r23 + g4*l45*r24 + g5*l55*r25   

    a31 = g1*l11*r31 + g2*l21*r32 + g3*l31*r33 + g4*l41*r34 + g5*l51*r35
    a32 = g1*l12*r31 + g2*l22*r32 + g3*l32*r33 + g4*l42*r34 + g5*l52*r35
    a33 = g1*l13*r31 + g2*l23*r32 + g3*l33*r33 + g4*l43*r34 + g5*l53*r35
    a34 = g1*l14*r31 + g2*l24*r32 + g3*l34*r33 + g4*l44*r34 + g5*l54*r35
    a35 = g1*l15*r31 + g2*l25*r32 + g3*l35*r33 + g4*l45*r34 + g5*l55*r35    

    a41 = g1*l11*r41 + g2*l21*r42 + g3*l31*r43 + g4*l41*r44 + g5*l51*r45
    a42 = g1*l12*r41 + g2*l22*r42 + g3*l32*r43 + g4*l42*r44 + g5*l52*r45
    a43 = g1*l13*r41 + g2*l23*r42 + g3*l33*r43 + g4*l43*r44 + g5*l53*r45
    a44 = g1*l14*r41 + g2*l24*r42 + g3*l34*r43 + g4*l44*r44 + g5*l54*r45
    a45 = g1*l15*r41 + g2*l25*r42 + g3*l35*r43 + g4*l45*r44 + g5*l55*r45    

    a51 = g1*l11*r51 + g2*l21*r52 + g3*l31*r53 + g4*l41*r54 + g5*l51*r55
    a52 = g1*l12*r51 + g2*l22*r52 + g3*l32*r53 + g4*l42*r54 + g5*l52*r55
    a53 = g1*l13*r51 + g2*l23*r52 + g3*l33*r53 + g4*l43*r54 + g5*l53*r55
    a54 = g1*l14*r51 + g2*l24*r52 + g3*l34*r53 + g4*l44*r54 + g5*l54*r55
    a55 = g1*l15*r51 + g2*l25*r52 + g3*l35*r53 + g4*l45*r54 + g5*l55*r55    

    
	q1 = uR(i,1)-uL(i,1)
    q2 = uR(i,2)-uL(i,2)
    q3 = uR(i,3)-uL(i,3)
    q4 = uR(i,4)-uL(i,4)
    q5 = uR(i,5)-uL(i,5)    
    
    cr(1) = q1*a11 + q2*a12 + q3*a13 + q4*a14 + q5*a15
    cr(2) = q1*a21 + q2*a22 + q3*a23 + q4*a24 + q5*a25
    cr(3) = q1*a31 + q2*a32 + q3*a33 + q4*a34 + q5*a35
    cr(4) = q1*a41 + q2*a42 + q3*a43 + q4*a44 + q5*a45
    cr(5) = q1*a51 + q2*a52 + q3*a53 + q4*a54 + q5*a55

    	!compute flux in x-direction
    	do m=1,5	
		rf(i,m)=0.5d0*(hR(i,m)+hL(i,m)) - 0.5d0*cr(m)     
    	end do
    
    
	end do



return
end

!-----------------------------------------------------------------------------------!
!Riemann Solver: Roe flux
!u:  cell centered data
!uL: weno reconstructed values at left state (positive)
!uR: weno reconstructed values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine roe_z(n,uL,uR,hL,hR,rf)
implicit none

integer::n,m,i
real*8 ::uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5),cr(5)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,hhLL,rhRR,uuRR,vvRR,eeRR,ppRR,hhRR,wwRR
real*8 ::uu,vv,ww,hh,aa,g1,g2,g3,g4,g5,q1,q2,q3,q4,q5
real*8 ::a11,a12,a13,a14,a15,a21,a22,a23,a24,a25,a31,a32,a33,a34,a35,a41,a42,a43,a44,a45
real*8 ::r11,r12,r13,r14,r15,r21,r22,r23,r24,r25,r31,r32,r33,r34,r35,r41,r42,r43,r44,r45
real*8 ::l11,l12,l13,l14,l15,l21,l22,l23,l24,l25,l31,l32,l33,l34,l35,l41,l42,l43,l44,l45
real*8 ::a51,a52,a53,a54,a55,r51,r52,r53,r54,r55,l51,l52,l53,l54,l55
real*8 ::capVs

common /fluids/ gamma

	!-----------------------------!
	!Roe flux in z-direction
    !-----------------------------!
	do i=0,n

	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
   	wwLL = uL(i,4)/rhLL	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    hhLL = eeLL + ppLL/rhLL

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    hhRR = eeRR + ppRR/rhRR


	!Roe averages

	uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
    ww = (dsqrt(rhLL)*wwLL + dsqrt(rhRR)*wwRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv+ww*ww)))

    !eigenvalues
    g1 = dabs(ww - aa)
    g2 = dabs(ww)
    g3 = dabs(ww)
    g4 = dabs(ww)
    g5 = dabs(ww + aa)

	!right eigenvectors
    r11 = 1.0d0
    r12 = 0.0d0
    r13 = 0.0d0
    r14 = 1.0d0
    r15 = 1.0d0

    r21 = uu
    r22 = 1.0d0
    r23 = 0.0d0
    r24 = uu
    r25 = uu    
    
	r31 = vv
    r32 = 0.0d0
    r33 = 1.0d0
    r34 = vv
    r35 = vv

	r41 = ww-aa
    r42 = 0.0d0
    r43 = 0.0d0
    r44 = ww
    r45 = ww+aa

    r51 = hh - ww*aa
	r52 = uu
    r53 = vv
    r54 = 0.5d0*(uu*uu+vv*vv+ww*ww)
    r55 = hh + ww*aa
	
	capVs = uu*uu+vv*vv+ww*ww

	!left eigenvectors
    l11 = (hh*ww+0.5d0*(capVs)*(aa-ww))/(2.0d0*aa*hh-aa*(capVs))
    l12 =-(aa*uu)/(2.0d0*aa*hh-aa*capVs)
    l13 =-(aa*vv)/(2.0d0*aa*hh-aa*capVs)
    l14 = (-hh-aa*ww+0.5d0*(capVs))/(2.0d0*aa*hh-aa*capVs)
    l15 = (aa)/(2.0d0*aa*hh-aa*capVs)

    l21 = (-2.0d0*hh+capVs)*(uu*aa)/(2.0d0*aa*hh-aa*capVs)
    l22 = 1.0d0
    l23 = 0.0d0
    l24 = 0.0d0
    l25 = 0.0d0
    
	l31 = (-2.0d0*hh+capVs)*(vv*aa)/(2.0d0*aa*hh-aa*capVs) 
    l32 = 0.0d0      
    l33 = 1.0d0      
    l34 = 0.0d0      
    l35 = 0.0d0      

	l41 = 2.0d0*aa*(hh-capVs)/(2.*aa*hh-aa*capVs)
    l42 = 2.0d0*aa*uu/(2.*aa*hh-aa*capVs)
    l43 = 2.0d0*aa*vv/(2.*aa*hh-aa*capVs)
    l44 = 2.0d0*aa*ww/(2.*aa*hh-aa*capVs)
    l45 =-2.0d0*aa/(2.*aa*hh-aa*capVs)

    l51 = (capVs*0.5d0*(ww+aa)-ww*hh)/(2.0d0*aa*hh-aa*(capVs))
	l52 =-aa*uu/(2.0d0*aa*hh-aa*(capVs))
    l53 =-aa*vv/(2.0d0*aa*hh-aa*(capVs)) 
    l54 = (hh-aa*ww-0.5d0*capVs)/(2.0d0*aa*hh-aa*(capVs))
    l55 = aa/(2.0d0*aa*hh-aa*(capVs))

    a11 = g1*l11*r11 + g2*l21*r12 + g3*l31*r13 + g4*l41*r14 + g5*l51*r15
    a12 = g1*l12*r11 + g2*l22*r12 + g3*l32*r13 + g4*l42*r14 + g5*l52*r15
    a13 = g1*l13*r11 + g2*l23*r12 + g3*l33*r13 + g4*l43*r14 + g5*l53*r15
    a14 = g1*l14*r11 + g2*l24*r12 + g3*l34*r13 + g4*l44*r14 + g5*l54*r15
    a15 = g1*l15*r11 + g2*l25*r12 + g3*l35*r13 + g4*l45*r14 + g5*l55*r15    

    a21 = g1*l11*r21 + g2*l21*r22 + g3*l31*r23 + g4*l41*r24 + g5*l51*r25
    a22 = g1*l12*r21 + g2*l22*r22 + g3*l32*r23 + g4*l42*r24 + g5*l52*r25
    a23 = g1*l13*r21 + g2*l23*r22 + g3*l33*r23 + g4*l43*r24 + g5*l53*r25
    a24 = g1*l14*r21 + g2*l24*r22 + g3*l34*r23 + g4*l44*r24 + g5*l54*r25
    a25 = g1*l15*r21 + g2*l25*r22 + g3*l35*r23 + g4*l45*r24 + g5*l55*r25   

    a31 = g1*l11*r31 + g2*l21*r32 + g3*l31*r33 + g4*l41*r34 + g5*l51*r35
    a32 = g1*l12*r31 + g2*l22*r32 + g3*l32*r33 + g4*l42*r34 + g5*l52*r35
    a33 = g1*l13*r31 + g2*l23*r32 + g3*l33*r33 + g4*l43*r34 + g5*l53*r35
    a34 = g1*l14*r31 + g2*l24*r32 + g3*l34*r33 + g4*l44*r34 + g5*l54*r35
    a35 = g1*l15*r31 + g2*l25*r32 + g3*l35*r33 + g4*l45*r34 + g5*l55*r35    

    a41 = g1*l11*r41 + g2*l21*r42 + g3*l31*r43 + g4*l41*r44 + g5*l51*r45
    a42 = g1*l12*r41 + g2*l22*r42 + g3*l32*r43 + g4*l42*r44 + g5*l52*r45
    a43 = g1*l13*r41 + g2*l23*r42 + g3*l33*r43 + g4*l43*r44 + g5*l53*r45
    a44 = g1*l14*r41 + g2*l24*r42 + g3*l34*r43 + g4*l44*r44 + g5*l54*r45
    a45 = g1*l15*r41 + g2*l25*r42 + g3*l35*r43 + g4*l45*r44 + g5*l55*r45    

    a51 = g1*l11*r51 + g2*l21*r52 + g3*l31*r53 + g4*l41*r54 + g5*l51*r55
    a52 = g1*l12*r51 + g2*l22*r52 + g3*l32*r53 + g4*l42*r54 + g5*l52*r55
    a53 = g1*l13*r51 + g2*l23*r52 + g3*l33*r53 + g4*l43*r54 + g5*l53*r55
    a54 = g1*l14*r51 + g2*l24*r52 + g3*l34*r53 + g4*l44*r54 + g5*l54*r55
    a55 = g1*l15*r51 + g2*l25*r52 + g3*l35*r53 + g4*l45*r54 + g5*l55*r55    

    
	q1 = uR(i,1)-uL(i,1)
    q2 = uR(i,2)-uL(i,2)
    q3 = uR(i,3)-uL(i,3)
    q4 = uR(i,4)-uL(i,4)
    q5 = uR(i,5)-uL(i,5)    
    
    cr(1) = q1*a11 + q2*a12 + q3*a13 + q4*a14 + q5*a15
    cr(2) = q1*a21 + q2*a22 + q3*a23 + q4*a24 + q5*a25
    cr(3) = q1*a31 + q2*a32 + q3*a33 + q4*a34 + q5*a35
    cr(4) = q1*a41 + q2*a42 + q3*a43 + q4*a44 + q5*a45
    cr(5) = q1*a51 + q2*a52 + q3*a53 + q4*a54 + q5*a55

    	!compute flux in x-direction
    	do m=1,5	
		rf(i,m)=0.5d0*(hR(i,m)+hL(i,m)) - 0.5d0*cr(m)     
    	end do
    
    
	end do


return
end

!-----------------------------------------------------------------------------------!
!Riemann Solver: HLL flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine hll_x(n,uL,uR,hL,hR,rf)
implicit none
integer::n,m,i
real*8 ::uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,aaLL,rhRR,uuRR,vvRR,wwRR,eeRR,ppRR,aaRr
real*8 ::SL,SR

common /fluids/ gamma


do i=0,n
  	
	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
	wwLL = uL(i,4)/rhLL    	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    aaRR = dsqrt(gamma*ppRR/rhRR)

	!Compute SL and SR
	!characteristics speed

		SL = min(uuLL,uuRR) - max(aaLL,aaRR)
		SR = max(uuLL,uuRR) + max(aaLL,aaRR)


	!compute HLL flux in x-direction
	if(SL.ge.0.0d0) then
		do m=1,5  
    	rf(i,m)=hL(i,m)     
		end do	
	else if (SR.le.0.0d0) then
		do m=1,5  
    	rf(i,m)=hR(i,m)     
		end do	
	else 
		do m=1,5  
    	rf(i,m)=(SR*hL(i,m)-SL*hR(i,m)+SL*SR*(uR(i,m)-uL(i,m)))/(SR-SL)     
		end do	
	end if
	
end do

return
end


!-----------------------------------------------------------------------------------!
!Riemann Solver: HLL flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine hll_y(n,uL,uR,hL,hR,rf)
implicit none
integer::n,m,i
real*8 ::uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,aaLL,rhRR,uuRR,vvRR,wwRR,eeRR,ppRR,aaRr
real*8 ::SL,SR

common /fluids/ gamma


do i=0,n
  	
	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
	wwLL = uL(i,4)/rhLL    	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    aaRR = dsqrt(gamma*ppRR/rhRR)

	!Compute SL and SR
	!characteristics speed

		SL = min(vvLL,vvRR) - max(aaLL,aaRR)
		SR = max(vvLL,vvRR) + max(aaLL,aaRR)


	!compute HLL flux in x-direction
	if(SL.ge.0.0d0) then
		do m=1,5  
    	rf(i,m)=hL(i,m)     
		end do	
	else if (SR.le.0.0d0) then
		do m=1,5  
    	rf(i,m)=hR(i,m)     
		end do	
	else 
		do m=1,5  
    	rf(i,m)=(SR*hL(i,m)-SL*hR(i,m)+SL*SR*(uR(i,m)-uL(i,m)))/(SR-SL)     
		end do	
	end if
	
end do

return
end

!-----------------------------------------------------------------------------------!
!Riemann Solver: HLL flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine hll_z(n,uL,uR,hL,hR,rf)
implicit none
integer::n,m,i
real*8 ::uL(0:n,5),uR(0:n,5),hL(0:n,5),hR(0:n,5),rf(0:n,5)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,aaLL,rhRR,uuRR,vvRR,wwRR,eeRR,ppRR,aaRr
real*8 ::SL,SR

common /fluids/ gamma

do i=0,n
  	
	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
	wwLL = uL(i,4)/rhLL    	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    aaRR = dsqrt(gamma*ppRR/rhRR)

	!Compute SL and SR
	!characteristics speed

		SL = min(wwLL,wwRR) - max(aaLL,aaRR)
		SR = max(wwLL,wwRR) + max(aaLL,aaRR)


	!compute HLL flux in x-direction
	if(SL.ge.0.0d0) then
		do m=1,5  
    	rf(i,m)=hL(i,m)     
		end do	
	else if (SR.le.0.0d0) then
		do m=1,5  
    	rf(i,m)=hR(i,m)     
		end do	
	else 
		do m=1,5  
    	rf(i,m)=(SR*hL(i,m)-SL*hR(i,m)+SL*SR*(uR(i,m)-uL(i,m)))/(SR-SL)     
		end do	
	end if
	
end do

return
end

!-----------------------------------------------------------------------------------!
!Riemann Solver: AUSM flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine ausm_x(n,uL,uR,rf)
implicit none
integer::n,i
real*8 ::uL(0:n,5),uR(0:n,5),rf(0:n,5)
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,hhLL,aaLL,rhRR,uuRR,vvRR,wwRR,eeRR,ppRR,hhRR,aaRR
real*8 ::MLL,pLL,MRR,pRR,M12,p12
real*8 ::gamma
    
common /fluids/ gamma

    !-----------------------------!
	!AUSM flux in x-direction
    !-----------------------------!

do i=0,n
  	
	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
	wwLL = uL(i,4)/rhLL    	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    hhLL = eeLL + ppLL/rhLL
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    hhRR = eeRR + ppRR/rhRR
    aaRR = dsqrt(gamma*ppRR/rhRR)
    
	if (dabs(uuLL/aaLL).le.1.0d0) then
	MLL = 0.25d0*(uuLL/aaLL + 1.0d0)**2
	pLL = 0.25d0*ppLL*((uuLL/aaLL + 1.0d0)**2)*(2.0d0-uuLL/aaLL)
	!pLL = 0.5d0*ppLL*(1.0d0+uuLL/aaLL)
	else
	MLL = 0.5d0*(uuLL/aaLL + dabs(uuLL/aaLL))
	pLL = 0.5d0*ppLL*(uuLL/aaLL + dabs(uuLL/aaLL))/(uuLL/aaLL)
	end if

	if (dabs(uuRR/aaRR).le.1.0d0) then
	MRR =-0.25d0*(uuRR/aaRR - 1.0d0)**2
	pRR = 0.25d0*ppRR*((uuRR/aaRR - 1.0d0)**2)*(2.0d0+uuRR/aaRR)
	!pRR = 0.5d0*ppRR*(1.0d0-uuRR/aaRR)
	else
	MRR = 0.5d0*(uuRR/aaRR - dabs(uuRR/aaRR))
	pRR = 0.5d0*ppRR*(uuRR/aaRR - dabs(uuRR/aaRR))/(uuRR/aaRR)
	end if

	M12 = MLL + MRR
	p12 = pLL + pRR

	!compute flux in x-direction
		
    rf(i,1)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR) + (M12+dabs(M12))*(rhLL*aaLL)) 
	rf(i,2)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*uuRR) + (M12+dabs(M12))*(rhLL*aaLL*uuLL)) + p12
	rf(i,3)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*vvRR) + (M12+dabs(M12))*(rhLL*aaLL*vvLL)) 
	rf(i,4)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*wwRR) + (M12+dabs(M12))*(rhLL*aaLL*wwLL))   
	rf(i,5)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*hhRR) + (M12+dabs(M12))*(rhLL*aaLL*hhLL))       
    
end do	

return
end


!-----------------------------------------------------------------------------------!
!Riemann Solver: AUSM flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine ausm_y(n,uL,uR,rf)
implicit none
integer::n,i
real*8 ::uL(0:n,5),uR(0:n,5),rf(0:n,5)
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,hhLL,aaLL,rhRR,uuRR,vvRR,wwRR,eeRR,ppRR,hhRR,aaRR
real*8 ::MLL,pLL,MRR,pRR,M12,p12
real*8 ::gamma
    
common /fluids/ gamma

    !-----------------------------!
	!AUSM flux in y-direction
    !-----------------------------!

do i=0,n
  	
	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
	wwLL = uL(i,4)/rhLL    	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    hhLL = eeLL + ppLL/rhLL
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    hhRR = eeRR + ppRR/rhRR
    aaRR = dsqrt(gamma*ppRR/rhRR)
    
	if (dabs(vvLL/aaLL).le.1.0d0) then
	MLL = 0.25d0*(vvLL/aaLL + 1.0d0)**2
	pLL = 0.25d0*ppLL*((vvLL/aaLL + 1.0d0)**2)*(2.0d0-vvLL/aaLL)
	!pLL = 0.5d0*ppLL*(1.0d0+uuLL/aaLL)
	else
	MLL = 0.5d0*(vvLL/aaLL + dabs(vvLL/aaLL))
	pLL = 0.5d0*ppLL*(vvLL/aaLL + dabs(vvLL/aaLL))/(vvLL/aaLL)
	end if

	if (dabs(vvRR/aaRR).le.1.0d0) then
	MRR =-0.25d0*(vvRR/aaRR - 1.0d0)**2
	pRR = 0.25d0*ppRR*((vvRR/aaRR - 1.0d0)**2)*(2.0d0+vvRR/aaRR)
	!pRR = 0.5d0*ppRR*(1.0d0-uuRR/aaRR)
	else
	MRR = 0.5d0*(vvRR/aaRR - dabs(vvRR/aaRR))
	pRR = 0.5d0*ppRR*(vvRR/aaRR - dabs(vvRR/aaRR))/(vvRR/aaRR)
	end if

	M12 = MLL + MRR
	p12 = pLL + pRR

	!compute flux in y-direction
		
    rf(i,1)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR) + (M12+dabs(M12))*(rhLL*aaLL)) 
	rf(i,2)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*uuRR) + (M12+dabs(M12))*(rhLL*aaLL*uuLL))
	rf(i,3)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*vvRR) + (M12+dabs(M12))*(rhLL*aaLL*vvLL)) + p12
	rf(i,4)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*wwRR) + (M12+dabs(M12))*(rhLL*aaLL*wwLL))   
	rf(i,5)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*hhRR) + (M12+dabs(M12))*(rhLL*aaLL*hhLL))       
    
end do	

return
end

!-----------------------------------------------------------------------------------!
!Riemann Solver: AUSM flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine ausm_z(n,uL,uR,rf)
implicit none
integer::n,i
real*8 ::uL(0:n,5),uR(0:n,5),rf(0:n,5)
real*8 ::rhLL,uuLL,vvLL,wwLL,eeLL,ppLL,hhLL,aaLL,rhRR,uuRR,vvRR,wwRR,eeRR,ppRR,hhRR,aaRR
real*8 ::MLL,pLL,MRR,pRR,M12,p12
real*8 ::gamma
    
common /fluids/ gamma

    !-----------------------------!
	!AUSM flux in z-direction
    !-----------------------------!

do i=0,n
  	
	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL
	wwLL = uL(i,4)/rhLL    	
	eeLL = uL(i,5)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL+wwLL*wwLL))
    hhLL = eeLL + ppLL/rhLL
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	wwRR = uR(i,4)/rhRR    
	eeRR = uR(i,5)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR+wwRR*wwRR))
    hhRR = eeRR + ppRR/rhRR
    aaRR = dsqrt(gamma*ppRR/rhRR)
    
	if (dabs(wwLL/aaLL).le.1.0d0) then
	MLL = 0.25d0*(wwLL/aaLL + 1.0d0)**2
	pLL = 0.25d0*ppLL*((wwLL/aaLL + 1.0d0)**2)*(2.0d0-wwLL/aaLL)
	!pLL = 0.5d0*ppLL*(1.0d0+uuLL/aaLL)
	else
	MLL = 0.5d0*(wwLL/aaLL + dabs(wwLL/aaLL))
	pLL = 0.5d0*ppLL*(wwLL/aaLL + dabs(wwLL/aaLL))/(wwLL/aaLL)
	end if

	if (dabs(wwRR/aaRR).le.1.0d0) then
	MRR =-0.25d0*(wwRR/aaRR - 1.0d0)**2
	pRR = 0.25d0*ppRR*((wwRR/aaRR - 1.0d0)**2)*(2.0d0+wwRR/aaRR)
	!pRR = 0.5d0*ppRR*(1.0d0-wwRR/aaRR)
	else
	MRR = 0.5d0*(wwRR/aaRR - dabs(wwRR/aaRR))
	pRR = 0.5d0*ppRR*(wwRR/aaRR - dabs(wwRR/aaRR))/(wwRR/aaRR)
	end if

	M12 = MLL + MRR
	p12 = pLL + pRR

	!compute flux in z-direction
		
    rf(i,1)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR) + (M12+dabs(M12))*(rhLL*aaLL)) 
	rf(i,2)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*uuRR) + (M12+dabs(M12))*(rhLL*aaLL*uuLL))
	rf(i,3)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*vvRR) + (M12+dabs(M12))*(rhLL*aaLL*vvLL))
	rf(i,4)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*wwRR) + (M12+dabs(M12))*(rhLL*aaLL*wwLL)) + p12   
	rf(i,5)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*hhRR) + (M12+dabs(M12))*(rhLL*aaLL*hhLL))       
    
end do	

return
end


!-----------------------------------------------------------------------------------!
!5th order WENO reconstruction
!-----------------------------------------------------------------------------------!
subroutine weno5(n,q,qL,qR)
implicit none
integer::n,m,i,pweno
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::eps,a,b,c,h,g,a1,a2,a3,w1,w2,w3,q1,q2,q3
real*8,allocatable ::b1(:),b2(:),b3(:)


common /weno_power/ pweno
common /weno_epsilon_j/ eps
!common /weno_epsilon_z/ epz
    
    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0
    
    a = 1.0d0/10.0d0
    b = 3.0d0/5.0d0
    c = 3.0d0/10.0d0

!compute beta
allocate(b1(0:n+1))
allocate(b2(0:n+1))
allocate(b3(0:n+1))

do m=1,5
  
	!Smoothness indicators
	do i=0,n+1

	b1(i) = h*(q(i-2,m)-2.0d0*q(i-1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i-2,m)-4.0d0*q(i-1,m)+3.0d0*q(i,m))**2

    b2(i) = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i-1,m)-q(i+1,m))**2
 
 	b3(i) = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
			+ 0.25d0*(3.0d0*q(i,m)-4.0d0*q(i+1,m)+q(i+2,m))**2

	end do


    
	do i=0,n
  
	!positive reconstruction at i+1/2

    a1 = a/(eps+b1(i))**pweno
    a2 = b/(eps+b2(i))**pweno
    a3 = c/(eps+b3(i))**pweno

	!a1 = a*(1.0d0 + (dabs(b1(i)-b3(i))/(epz+b1(i)))**pweno)
	!a2 = b*(1.0d0 + (dabs(b1(i)-b3(i))/(epz+b2(i)))**pweno)
	!a3 = c*(1.0d0 + (dabs(b1(i)-b3(i))/(epz+b3(i)))**pweno)

    
    w1 = a1/(a1+a2+a3)
    w2 = a2/(a1+a2+a3)
    w3 = a3/(a1+a2+a3)
    
	q1 = g*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))
	q2 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q3 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))

		
	qL(i,m) = w1*q1 + w2*q2 + w3*q3


    !negative reconstruction at i+1/2
  
    a1 = c/(eps+b1(i+1))**pweno
    a2 = b/(eps+b2(i+1))**pweno
    a3 = a/(eps+b3(i+1))**pweno

	!a1 = c*(1.0d0 + (dabs(b1(i+1)-b3(i+1))/(epz+b1(i+1)))**pweno)
	!a2 = b*(1.0d0 + (dabs(b1(i+1)-b3(i+1))/(epz+b2(i+1)))**pweno)
	!a3 = a*(1.0d0 + (dabs(b1(i+1)-b3(i+1))/(epz+b3(i+1)))**pweno)
    
    w1 = a1/(a1+a2+a3)
    w2 = a2/(a1+a2+a3)
    w3 = a3/(a1+a2+a3)
    
	q1 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q2 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))
	q3 = g*(11.0d0*q(i+1,m)-7.0d0*q(i+2,m)+ 2.0d0*q(i+3,m))

    qR(i,m) = w1*q1 + w2*q2 + w3*q3
           
 		     
	end do
    
end do


deallocate(b1,b2,b3)

return 
end

!-----------------------------------------------------------------------------------!
!5th order upwind scheme
!-----------------------------------------------------------------------------------!
subroutine upwind5(n,q,qL,qR)
implicit none
integer::n,m,i,ihyb
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::gg

gg = 1.0d0/60.0d0
         
do i=0,n
      
	do m=1,5
      
        qL(i,m) = gg*(2.0d0*q(i-2,m)-13.0d0*q(i-1,m) &
                 +47.0d0*q(i,m)+27.0d0*q(i+1,m)-3.0d0*q(i+2,m))
   
        qR(i,m) = gg*(2.0d0*q(i+3,m)-13.0d0*q(i+2,m) &
                 +47.0d0*q(i+1,m)+27.0d0*q(i,m)-3.0d0*q(i-1,m))
            
  	end do
end do
    

    

return
end

!-----------------------------------------------------------------------------------!
!Central scheme 6th order
!-----------------------------------------------------------------------------------!
subroutine central6(n,q,qL,qR)
implicit none
integer::n,m,i,ihyb
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::gg

gg = 1.0d0/60.0d0
         
do i=0,n
      
	do m=1,5
      
        qL(i,m) = gg*(q(i-2,m)-8.0d0*q(i-1,m)+37.0d0*q(i,m) &
                 +37.0d0*q(i+1,m)-8.0d0*q(i+2,m) + q(i+3,m))
   
        qR(i,m) = gg*(q(i+3,m)-8.0d0*q(i+2,m)+37.0d0*q(i+1,m) &
                 +37.0d0*q(i,m)-8.0d0*q(i-1,m) + q(i-2,m))
            
  	end do
end do


return
end

!-----------------------------------------------------------------------------------!
!DRP 4th order
!-----------------------------------------------------------------------------------!
subroutine drp4(n,q,qL,qR)
implicit none
integer::n,m,i,ihyb
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::aa,bb,cc,dd,ee,ff,disp,diss

!optimal dispersion coefficients:
!disp = 11.0d0/63.0d0 !v=0
!disp = 0.136203d0 !v=1
!disp = 0.107065d0 !v=2
!disp = 0.0860377d0 !v=3
!disp = 0.0714071d0 !v=4
!disp = 0.0545455d0 !v=6
disp = 0.0463783d0 !v=8
!disp = 0.0420477d0 !v=10
!disp = 1.0d0/30.0d0  !v-->infinity

!dissipation control for DRP (default=0.01)
diss = 0.01d0

!if disp=1/30 and diss=0 ==> central6 scheme

	aa = 0.5d0*(disp+diss)
    bb =-1.0d0/12.0d0 - 1.5d0*disp - 2.5d0*diss
    cc = 7.0d0/12.0d0 + disp + 5.0d0*diss
    dd = 7.0d0/12.0d0 + disp - 5.0d0*diss
    ee =-1.0d0/12.0d0 - 1.5d0*disp + 2.5d0*diss
    ff = 0.5d0*(disp-diss)
    

do i=0,n
      
	do m=1,5
      
        qL(i,m) = aa*q(i-2,m) + bb*q(i-1,m) + cc*q(i,m) &
                 +dd*q(i+1,m) + ee*q(i+2,m) + ff*q(i+3,m)
   
        qR(i,m) = aa*q(i+3,m) + bb*q(i+2,m) + cc*q(i+1,m) &
                 +dd*q(i,m) + ee*q(i-1,m) + ff*q(i-2,m)
            
  	end do
end do



return
end



!-----------------------------------------------------------------------------------!
!Bandwidth optimized upwind scheme (4th-order)
!-----------------------------------------------------------------------------------!
subroutine bandwidth4(n,q,qL,qR)
implicit none
integer::n,m,i,ihyb
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::aa,bb,cc,dd,ee,ff,d1,d2,d3,d4

    !linear coefficients
    d1 = 0.094647545896d0
    d2 = 0.428074212384d0
    d3 = 0.408289331408d0
    d4 = 0.068988910311d0
    
	aa = 1.0d0/3.0d0*d1
    bb =-7.0d0/6.0d0*d1 - 1.0d0/6.0d0*d2
    cc = 11.0d0/6.0d0*d1 + 5.0d0/6.0d0*d2 + 1.0d0/3.0d0*d3
    dd = 1.0d0/3.0d0*d2 + 5.0d0/6.0d0*d3 + 11.0d0/6.0d0*d4
    ee =-1.0d0/6.0d0*d3 - 7.0d0/6.0d0*d4
    ff = 1.0d0/3.0d0*d4

    
do i=0,n
      
	do m=1,5
      
        qL(i,m) = aa*q(i-2,m) + bb*q(i-1,m) + cc*q(i,m) &
                 +dd*q(i+1,m) + ee*q(i+2,m) + ff*q(i+3,m)
   
        qR(i,m) = aa*q(i+3,m) + bb*q(i+2,m) + cc*q(i+1,m) &
                 +dd*q(i,m) + ee*q(i-1,m) + ff*q(i-2,m)
            
  	end do
end do


return
end
  
!-----------------------------------------------------------------------------------!
!5th order WENO reconstruction (jiang-shu)
!Classical WENO scheme
!recommended eps=1.0d-6, pweno=2
!-----------------------------------------------------------------------------------!
subroutine jweno5(n,q,qL,qR)
implicit none
integer::n,m,i,pweno
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::eps,d1,d2,d3,h,g,a1,a2,a3,w1,w2,w3,q1,q2,q3
real*8 ::b1,b2,b3

common /weno_power/ pweno
common /weno_epsilon_j/ eps

            
    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0
    
    d1 = 1.0d0/10.0d0
    d2 = 3.0d0/5.0d0
    d3 = 3.0d0/10.0d0


do m=1,5
  
    !positive reconstruction at i+1/2
    
	do i=0,n

    b1 = h*(q(i-2,m)-2.0d0*q(i-1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i-2,m)-4.0d0*q(i-1,m)+3.0d0*q(i,m))**2

    b2 = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i-1,m)-q(i+1,m))**2
 
 	b3 = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
			+ 0.25d0*(3.0d0*q(i,m)-4.0d0*q(i+1,m)+q(i+2,m))**2
            
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno

    w1 = a1/(a1+a2+a3)
    w2 = a2/(a1+a2+a3)
    w3 = a3/(a1+a2+a3)
    
	q1 = g*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))
	q2 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q3 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))

		
	qL(i,m) = w1*q1 + w2*q2 + w3*q3
 	
    end do



    !negative reconstruction at i+1/2
    do i=0,n

    b1 = h*(q(i+3,m)-2.0d0*q(i+2,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i+3,m)-4.0d0*q(i+2,m)+3.0d0*q(i+1,m))**2

    b2 = h*(q(i+2,m)-2.0d0*q(i+1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i+2,m)-q(i,m))**2
 
 	b3 = h*(q(i+1,m)-2.0d0*q(i,m)+q(i-1,m))**2 &
			+ 0.25d0*(3.0d0*q(i+1,m)-4.0d0*q(i,m)+q(i-1,m))**2

            
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno

    w1 = a1/(a1+a2+a3)
    w2 = a2/(a1+a2+a3)
    w3 = a3/(a1+a2+a3)
    
	q1 = g*(2.0d0*q(i+3,m)-7.0d0*q(i+2,m)+11.0d0*q(i+1,m))
	q2 = g*(-q(i+2,m)+5.0d0*q(i+1,m)+2.0d0*q(i,m))
	q3 = g*(2.0d0*q(i+1,m)+5.0d0*q(i,m)-q(i-1,m))
    
    qR(i,m) = w1*q1 + w2*q2 + w3*q3
           		     
	end do
	
    
end do


return 
end


    
!-----------------------------------------------------------------------------------!
!5th order WENO reconstruction (shen-zha)
!This is less dissipative compared to original Jiang and Shu's WENO
!-----------------------------------------------------------------------------------!
subroutine zweno5(n,q,qL,qR)
implicit none
integer::n,m,i,pweno
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::epz,d1,d2,d3,h,g,a1,a2,a3,w1,w2,w3,q1,q2,q3
real*8 ::b1,b2,b3

common /weno_power/ pweno
common /weno_epsilon_z/ epz
    
    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0
    
    d1 = 1.0d0/10.0d0
    d2 = 3.0d0/5.0d0
    d3 = 3.0d0/10.0d0


do m=1,5
  
    !positive reconstruction at i+1/2
    
	do i=0,n

    b1 = h*(q(i-2,m)-2.0d0*q(i-1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i-2,m)-4.0d0*q(i-1,m)+3.0d0*q(i,m))**2

    b2 = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i-1,m)-q(i+1,m))**2
 
 	b3 = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
			+ 0.25d0*(3.0d0*q(i,m)-4.0d0*q(i+1,m)+q(i+2,m))**2
 
    !a1 = d1/(eps+b1)**pweno
    !a2 = d2/(eps+b2)**pweno
    !a3 = d3/(eps+b3)**pweno
    
	a1 = d1*(1.0d0 + (dabs(b1-b3)/(epz+b1))**pweno)
	a2 = d2*(1.0d0 + (dabs(b1-b3)/(epz+b2))**pweno)
	a3 = d3*(1.0d0 + (dabs(b1-b3)/(epz+b3))**pweno)
           
    w1 = a1/(a1+a2+a3)
    w2 = a2/(a1+a2+a3)
    w3 = a3/(a1+a2+a3)
    
	q1 = g*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))
	q2 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q3 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))

		
	qL(i,m) = w1*q1 + w2*q2 + w3*q3
 	
    end do

    

    !negative reconstruction at i+1/2
    do i=0,n

    b1 = h*(q(i+3,m)-2.0d0*q(i+2,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i+3,m)-4.0d0*q(i+2,m)+3.0d0*q(i+1,m))**2

    b2 = h*(q(i+2,m)-2.0d0*q(i+1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i+2,m)-q(i,m))**2
 
 	b3 = h*(q(i+1,m)-2.0d0*q(i,m)+q(i-1,m))**2 &
			+ 0.25d0*(3.0d0*q(i+1,m)-4.0d0*q(i,m)+q(i-1,m))**2

            
    !a1 = d1/(eps+b1)**pweno
    !a2 = d2/(eps+b2)**pweno
    !a3 = d3/(eps+b3)**pweno

    a1 = d1*(1.0d0 + (dabs(b1-b3)/(epz+b1))**pweno)
	a2 = d2*(1.0d0 + (dabs(b1-b3)/(epz+b2))**pweno)
	a3 = d3*(1.0d0 + (dabs(b1-b3)/(epz+b3))**pweno)
    

    w1 = a1/(a1+a2+a3)
    w2 = a2/(a1+a2+a3)
    w3 = a3/(a1+a2+a3)
    
	q1 = g*(2.0d0*q(i+3,m)-7.0d0*q(i+2,m)+11.0d0*q(i+1,m))
	q2 = g*(-q(i+2,m)+5.0d0*q(i+1,m)+2.0d0*q(i,m))
	q3 = g*(2.0d0*q(i+1,m)+5.0d0*q(i,m)-q(i-1,m))
    
    qR(i,m) = w1*q1 + w2*q2 + w3*q3
           		     
	end do

     
end do


return 
end



!-----------------------------------------------------------------------------------!
!6th order WENO reconstruction (order-optimized central)
!-----------------------------------------------------------------------------------!
subroutine cweno5(n,q,qL,qR)
implicit none
integer::n,m,i,pweno
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::eps,d1,d2,d3,d4,h,g,a1,a2,a3,a4,w1,w2,w3,w4,q1,q2,q3,q4
real*8 ::b1,b2,b3,b4

common /weno_power/ pweno
common /weno_epsilon_j/ eps
!common /weno_epsilon_z/ epz
    
    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0
    
    !linear coefficients
    d1 = 1.0d0/20.0d0
    d2 = 9.0d0/20.0d0
    d3 = 9.0d0/20.0d0
    d4 = 1.0d0/20.0d0

do m=1,5
  
    !positive reconstruction at i+1/2
	do i=0,n
  
    !Smoothness indicators
	b1 = h*(q(i-2,m)-2.0d0*q(i-1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i-2,m)-4.0d0*q(i-1,m)+3.0d0*q(i,m))**2

    b2 = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i-1,m)-q(i+1,m))**2
 
 	b3 = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
			+ 0.25d0*(3.0d0*q(i,m)-4.0d0*q(i+1,m)+q(i+2,m))**2

 	b4 = h*(q(i+1,m)-2.0d0*q(i+2,m)+q(i+3,m))**2 &
			+ 0.25d0*(5.0d0*q(i+1,m)-8.0d0*q(i+2,m)+3.0d0*q(i+3,m))**2

	!limit the downwind indicator
    b2 = max(b1,b2)
    b3 = max(b2,b3)
    b4 = max(b3,b4)
    
	!nonlinear weighting
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno
    a4 = d4/(eps+b4)**pweno

	!normalized nonlinear weights
    w1 = a1/(a1+a2+a3+a4)
    w2 = a2/(a1+a2+a3+a4)
    w3 = a3/(a1+a2+a3+a4)
    w4 = a4/(a1+a2+a3+a4)
    
	q1 = g*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))
	q2 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q3 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))
    q4 = g*(11.0d0*q(i+1,m)-7.0d0*q(i+2,m)+2.0d0*q(i+3,m))
    
		
	qL(i,m) = w1*q1 + w2*q2 + w3*q3 + w4*q4

    end do

    
    !negative reconstruction at i+1/2
    do i=0,n

    !Smoothness indicators
	b1 = h*(q(i+3,m)-2.0d0*q(i+2,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i+3,m)-4.0d0*q(i+2,m)+3.0d0*q(i+1,m))**2

    b2 = h*(q(i+2,m)-2.0d0*q(i+1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i+2,m)-q(i,m))**2
 
 	b3 = h*(q(i+1,m)-2.0d0*q(i,m)+q(i-1,m))**2 &
			+ 0.25d0*(3.0d0*q(i+1,m)-4.0d0*q(i,m)+q(i-1,m))**2

 	b4 = h*(q(i,m)-2.0d0*q(i-1,m)+q(i-2,m))**2 &
			+ 0.25d0*(5.0d0*q(i,m)-8.0d0*q(i-1,m)+3.0d0*q(i-2,m))**2

	!limit the downwind indicator
    b2 = max(b1,b2)
    b3 = max(b2,b3)
    b4 = max(b3,b4)
  
    
	!nonlinear weighting
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno
    a4 = d4/(eps+b4)**pweno

	!normalized nonlinear weights
    w1 = a1/(a1+a2+a3+a4)
    w2 = a2/(a1+a2+a3+a4)
    w3 = a3/(a1+a2+a3+a4)
    w4 = a4/(a1+a2+a3+a4)
    
	q1 = g*(2.0d0*q(i+3,m)-7.0d0*q(i+2,m)+11.0d0*q(i+1,m))
	q2 = g*(-q(i+2,m)+5.0d0*q(i+1,m)+2.0d0*q(i,m))
	q3 = g*(2.0d0*q(i+1,m)+5.0d0*q(i,m)-q(i-1,m))
    q4 = g*(11.0d0*q(i,m)-7.0d0*q(i-1,m)+2.0d0*q(i-2,m))
    
		
	qR(i,m) = w1*q1 + w2*q2 + w3*q3 + w4*q4
    		     
	end do
    
end do


return 
end

!-----------------------------------------------------------------------------------!
!4th order WENO reconstruction (bandwidth-optimized) Martin et.al. Princeton
!-----------------------------------------------------------------------------------!
subroutine bweno5(n,q,qL,qR)
implicit none
integer::n,m,i,pweno
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::eps,d1,d2,d3,d4,h,g,a1,a2,a3,a4,w1,w2,w3,w4,q1,q2,q3,q4
real*8 ::b1,b2,b3,b4

common /weno_power/ pweno
common /weno_epsilon_j/ eps
!common /weno_epsilon_z/ epz
    
    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0
    
    !linear coefficients
    d1 = 0.094647545896d0
    d2 = 0.428074212384d0
    d3 = 0.408289331408d0
    d4 = 0.068988910311d0

do m=1,5
  
    !positive reconstruction at i+1/2
	do i=0,n
  
    !Smoothness indicators
	b1 = h*(q(i-2,m)-2.0d0*q(i-1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i-2,m)-4.0d0*q(i-1,m)+3.0d0*q(i,m))**2

    b2 = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i-1,m)-q(i+1,m))**2
 
 	b3 = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
			+ 0.25d0*(3.0d0*q(i,m)-4.0d0*q(i+1,m)+q(i+2,m))**2

 	b4 = h*(q(i+1,m)-2.0d0*q(i+2,m)+q(i+3,m))**2 &
			+ 0.25d0*(5.0d0*q(i+1,m)-8.0d0*q(i+2,m)+3.0d0*q(i+3,m))**2

	!limit the downwind indicator
    b2 = max(b1,b2)
    b3 = max(b2,b3)
    b4 = max(b3,b4)
    
	!nonlinear weighting
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno
    a4 = d4/(eps+b4)**pweno

	!normalized nonlinear weights
    w1 = a1/(a1+a2+a3+a4)
    w2 = a2/(a1+a2+a3+a4)
    w3 = a3/(a1+a2+a3+a4)
    w4 = a4/(a1+a2+a3+a4)
    
	q1 = g*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))
	q2 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q3 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))
    q4 = g*(11.0d0*q(i+1,m)-7.0d0*q(i+2,m)+2.0d0*q(i+3,m))
    
		
	qL(i,m) = w1*q1 + w2*q2 + w3*q3 + w4*q4

    end do

    
    !negative reconstruction at i+1/2
    do i=0,n

    !Smoothness indicators
	b1 = h*(q(i+3,m)-2.0d0*q(i+2,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i+3,m)-4.0d0*q(i+2,m)+3.0d0*q(i+1,m))**2

    b2 = h*(q(i+2,m)-2.0d0*q(i+1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i+2,m)-q(i,m))**2
 
 	b3 = h*(q(i+1,m)-2.0d0*q(i,m)+q(i-1,m))**2 &
			+ 0.25d0*(3.0d0*q(i+1,m)-4.0d0*q(i,m)+q(i-1,m))**2

 	b4 = h*(q(i,m)-2.0d0*q(i-1,m)+q(i-2,m))**2 &
			+ 0.25d0*(5.0d0*q(i,m)-8.0d0*q(i-1,m)+3.0d0*q(i-2,m))**2

	!limit the downwind indicator
    b2 = max(b1,b2)
    b3 = max(b2,b3)
    b4 = max(b3,b4)
  
    
	!nonlinear weighting
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno
    a4 = d4/(eps+b4)**pweno

	!normalized nonlinear weights
    w1 = a1/(a1+a2+a3+a4)
    w2 = a2/(a1+a2+a3+a4)
    w3 = a3/(a1+a2+a3+a4)
    w4 = a4/(a1+a2+a3+a4)
    
	q1 = g*(2.0d0*q(i+3,m)-7.0d0*q(i+2,m)+11.0d0*q(i+1,m))
	q2 = g*(-q(i+2,m)+5.0d0*q(i+1,m)+2.0d0*q(i,m))
	q3 = g*(2.0d0*q(i+1,m)+5.0d0*q(i,m)-q(i-1,m))
    q4 = g*(11.0d0*q(i,m)-7.0d0*q(i-1,m)+2.0d0*q(i-2,m))
    
		
	qR(i,m) = w1*q1 + w2*q2 + w3*q3 + w4*q4
    		     
	end do
    
end do

return 
end

!-----------------------------------------------------------------------------------!
!4th order WENO reconstruction (central-DRP)
!-----------------------------------------------------------------------------------!
subroutine dweno5(n,q,qL,qR)
implicit none
integer::n,m,i,pweno
real*8 ::q(-2:n+3,5),qL(0:n,5),qR(0:n,5)
real*8 ::eps,d1,d2,d3,d4,h,g,a1,a2,a3,a4,w1,w2,w3,w4,q1,q2,q3,q4
real*8 ::b1,b2,b3,b4,disp,diss

common /weno_power/ pweno
common /weno_epsilon_j/ eps
!common /weno_epsilon_z/ epz


!optimal dispersion coefficients:
!disp = 11.0d0/63.0d0 !v=0
!disp = 0.136203d0 !v=1
!disp = 0.107065d0 !v=2
!disp = 0.0860377d0 !v=3
!disp = 0.0714071d0 !v=4
!disp = 0.0545455d0 !v=6
disp = 0.0463783d0 !v=8
!disp = 0.0420477d0 !v=10
!disp = 1.0d0/30.0d0  !v-->infinity

!dissipation control for DRP (default=0.01)
diss = 0.01d0

    
    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0
    
    !linear coefficients
    d1 = 1.5d0*(disp+diss)
    d2 = 0.5d0*(1.0d0-3.0d0*disp+9.0d0*diss)
    d3 = 0.5d0*(1.0d0-3.0d0*disp-9.0d0*diss)
    d4 = 1.5d0*(disp-diss)

do m=1,5
  
    !positive reconstruction at i+1/2
	do i=0,n
  
    !Smoothness indicators
	b1 = h*(q(i-2,m)-2.0d0*q(i-1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i-2,m)-4.0d0*q(i-1,m)+3.0d0*q(i,m))**2

    b2 = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i-1,m)-q(i+1,m))**2
 
 	b3 = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
			+ 0.25d0*(3.0d0*q(i,m)-4.0d0*q(i+1,m)+q(i+2,m))**2

 	b4 = h*(q(i+1,m)-2.0d0*q(i+2,m)+q(i+3,m))**2 &
			+ 0.25d0*(5.0d0*q(i+1,m)-8.0d0*q(i+2,m)+3.0d0*q(i+3,m))**2

    !limit the downwind indicator
    b2 = max(b1,b2)
    b3 = max(b2,b3)
    b4 = max(b3,b4)

    
	!nonlinear weighting
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno
    a4 = d4/(eps+b4)**pweno

	!normalized nonlinear weights
    w1 = a1/(a1+a2+a3+a4)
    w2 = a2/(a1+a2+a3+a4)
    w3 = a3/(a1+a2+a3+a4)
    w4 = a4/(a1+a2+a3+a4)
    
	q1 = g*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))
	q2 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q3 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))
    q4 = g*(11.0d0*q(i+1,m)-7.0d0*q(i+2,m)+2.0d0*q(i+3,m))
    
		
	qL(i,m) = w1*q1 + w2*q2 + w3*q3 + w4*q4

    end do

    
    !negative reconstruction at i+1/2
    do i=0,n

    !Smoothness indicators
	b1 = h*(q(i+3,m)-2.0d0*q(i+2,m)+q(i+1,m))**2 &
			+ 0.25d0*(q(i+3,m)-4.0d0*q(i+2,m)+3.0d0*q(i+1,m))**2

    b2 = h*(q(i+2,m)-2.0d0*q(i+1,m)+q(i,m))**2 &
			+ 0.25d0*(q(i+2,m)-q(i,m))**2
 
 	b3 = h*(q(i+1,m)-2.0d0*q(i,m)+q(i-1,m))**2 &
			+ 0.25d0*(3.0d0*q(i+1,m)-4.0d0*q(i,m)+q(i-1,m))**2

 	b4 = h*(q(i,m)-2.0d0*q(i-1,m)+q(i-2,m))**2 &
			+ 0.25d0*(5.0d0*q(i,m)-8.0d0*q(i-1,m)+3.0d0*q(i-2,m))**2

    !limit the downwind indicator
    b2 = max(b1,b2)
    b3 = max(b2,b3)
    b4 = max(b3,b4)

    
	!nonlinear weighting
    a1 = d1/(eps+b1)**pweno
    a2 = d2/(eps+b2)**pweno
    a3 = d3/(eps+b3)**pweno
    a4 = d4/(eps+b4)**pweno

	!normalized nonlinear weights
    w1 = a1/(a1+a2+a3+a4)
    w2 = a2/(a1+a2+a3+a4)
    w3 = a3/(a1+a2+a3+a4)
    w4 = a4/(a1+a2+a3+a4)
    
	q1 = g*(2.0d0*q(i+3,m)-7.0d0*q(i+2,m)+11.0d0*q(i+1,m))
	q2 = g*(-q(i+2,m)+5.0d0*q(i+1,m)+2.0d0*q(i,m))
	q3 = g*(2.0d0*q(i+1,m)+5.0d0*q(i,m)-q(i-1,m))
    q4 = g*(11.0d0*q(i,m)-7.0d0*q(i-1,m)+2.0d0*q(i-2,m))
    
		
	qR(i,m) = w1*q1 + w2*q2 + w3*q3 + w4*q4
    		     
	end do
    
end do

return 
end




!-----------------------------------------------------------------------------------!
!Time Step
!-----------------------------------------------------------------------------------!
subroutine timestep(nx,ny,nz,dx,dy,dz,cfl,dt,q)
implicit none
integer::nx,ny,nz,i,j,k
real*8 ::dt,cfl,gamma,dx,dy,dz,smx,smy,smz,radx,rady,radz,p,a,l1,l2,l3
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)

common /fluids/ gamma

!Spectral radius of Jacobian
smx = 0.0d0
smy = 0.0d0
smz = 0.0d0

do k=1,nz
do j=1,ny
do i=1,nx

p = (gamma-1.0d0)*(q(i,j,k,5)-0.5d0*(q(i,j,k,2)*q(i,j,k,2)/q(i,j,k,1) &
                                    +q(i,j,k,3)*q(i,j,k,3)/q(i,j,k,1) &
                                    +q(i,j,k,4)*q(i,j,k,4)/q(i,j,k,1) ))
a = dsqrt(gamma*p/q(i,j,k,1)) 
 
!in-x direction  
l1=dabs(q(i,j,k,2)/q(i,j,k,1))
l2=dabs(q(i,j,k,2)/q(i,j,k,1) + a)
l3=dabs(q(i,j,k,2)/q(i,j,k,1) - a)
radx = max(l1,l2,l3)

!in-y direction  
l1=dabs(q(i,j,k,3)/q(i,j,k,1))
l2=dabs(q(i,j,k,3)/q(i,j,k,1) + a)
l3=dabs(q(i,j,k,3)/q(i,j,k,1) - a)
rady = max(l1,l2,l3)

!in -z direction
l1=dabs(q(i,j,k,4)/q(i,j,k,1))
l2=dabs(q(i,j,k,4)/q(i,j,k,1) + a)
l3=dabs(q(i,j,k,4)/q(i,j,k,1) - a)
radz = max(l1,l2,l3)

if (radx.gt.smx) smx = radx
if (rady.gt.smy) smy = rady
if (radz.gt.smz) smz = radz
    
end do
end do
end do

dt = min(cfl*dx/smx,cfl*dy/smy,cfl*dz/smz)

return 
end

!-----------------------------------------------------------------------------------!
!History: compute domain integrated total energy
!-----------------------------------------------------------------------------------!
subroutine history(nx,ny,nz,q,te)
implicit none
integer::nx,ny,nz,i,j,k
real*8 ::u,v,w,te
real*8 ::q(-2:nx+3,-2:ny+3,-2:nz+3,5)

te = 0.0d0
do k=1,nz
do j=1,ny
do i=1,nx
u=q(i,j,k,2)/q(i,j,k,1)
v=q(i,j,k,3)/q(i,j,k,1)
w=q(i,j,k,4)/q(i,j,k,1)
te = te + 0.5d0*(u*u + v*v + w*w)
end do
end do
end do

return 
end



!-----------------------------------------------------------------------------------!
!Computing x-fluxes from conserved quantities 
!-----------------------------------------------------------------------------------!
subroutine xflux(nx,u,f)
implicit none
integer::nx,i
real*8::gamma,p
real*8::u(0:nx,5),f(0:nx,5)

common /fluids/ gamma

do i=0,nx
p = (gamma-1.0d0)*(u(i,5)-0.5d0*(u(i,2)*u(i,2)/u(i,1) &
                                +u(i,3)*u(i,3)/u(i,1) &
                                +u(i,4)*u(i,4)/u(i,1) ) )
f(i,1) = u(i,2)
f(i,2) = u(i,2)*u(i,2)/u(i,1) + p
f(i,3) = u(i,2)*u(i,3)/u(i,1)
f(i,4) = u(i,2)*u(i,4)/u(i,1)
f(i,5) = (u(i,5)+ p)*u(i,2)/u(i,1)

end do

return
end

!-----------------------------------------------------------------------------------!
!Computing y-fluxes from conserved quantities
!-----------------------------------------------------------------------------------!
subroutine yflux(ny,u,g)
implicit none
integer::ny,j
real*8::gamma,p
real*8::u(0:ny,5),g(0:ny,5)

common /fluids/ gamma

do j=0,ny
p = (gamma-1.0d0)*(u(j,5)-0.5d0*(u(j,2)*u(j,2)/u(j,1) &
                                +u(j,3)*u(j,3)/u(j,1) &
                                +u(j,4)*u(j,4)/u(j,1) ) )
g(j,1) = u(j,3)
g(j,2) = u(j,3)*u(j,2)/u(j,1) 
g(j,3) = u(j,3)*u(j,3)/u(j,1) + p
g(j,4) = u(j,3)*u(j,4)/u(j,1)
g(j,5) = (u(j,5)+ p)*u(j,3)/u(j,1)
end do

return
end

!-----------------------------------------------------------------------------------!
!Computing z-fluxes from conserved quantities
!-----------------------------------------------------------------------------------!
subroutine zflux(nz,u,h)
implicit none
integer::nz,k
real*8::gamma,p
real*8::u(0:nz,5),h(0:nz,5)

common /fluids/ gamma

do k=0,nz
p = (gamma-1.0d0)*(u(k,5)-0.5d0*(u(k,2)*u(k,2)/u(k,1) &
                                +u(k,3)*u(k,3)/u(k,1) &
                                +u(k,4)*u(k,4)/u(k,1) ) )
h(k,1) = u(k,4)
h(k,2) = u(k,4)*u(k,2)/u(k,1) 
h(k,3) = u(k,4)*u(k,3)/u(k,1)
h(k,4) = u(k,4)*u(k,4)/u(k,1) + p
h(k,5) = (u(k,5)+ p)*u(k,4)/u(k,1)
end do

return
end
