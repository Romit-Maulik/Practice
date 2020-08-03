
!------------------------------------------------------------------------------!
!>>> Euler Solver in 2D Cartesian Domain >>>
!------------------------------------------------------------------------------!
!Parallel Code Implementation with MPI
!To run in Cowboy Cluster (OSU): 
!          module load openmpi-1.4/intel
!          mpif90 euler2d_MPI.f90
!		   qsub mexpress.job (mbatch.job or mlong.job)
!------------------------------------------------------------------------------!
!Schemes: 5th order WENO reconstruction 
!         Rusanov Riemann solver at interface fluxes
!         3rd order TVD RK for time integration
!------------------------------------------------------------------------------!
!Domain: In a unit square [x:-0.5,0.5]*[y:-0.5,0.5]
!BCs: Periodic boundary conditions for the KHI problem 
!     Open (transmissive) boundary conditions for the Riemann problem
!------------------------------------------------------------------------------!
!Cases: Kelvin-Helmholtz instability (KHI) problem (Ref1)
!       Riemann (SHOCK) problem (Ref 2)
!       Rayleigh-Taylor instability (RTI)
!------------------------------------------------------------------------------!
!References: 
!1-) O. San, K. Kara, Computers & Fluids 117 (2015) 24-41
!    Evaluation of Riemann flux solvers for WENO reconstruction schemes: 
!    Kelvin-Helmholtz instability
!2-) O. San, K. Kara, Computers & Fluids 89 (2014) 254-276
!    Numerical assessments of high-order accurate shock capturing schemes:
!    Kelvin-Helmholtz type vortical structures in high-resolutions
!------------------------------------------------------------------------------!
!Omer San/Romit Maulik
!Oklahoma State University, Stillwater
!CFDLab.org, cfdlab.osu@gmail.com
!Writen: June 25, 2016
!Updated: Jan 24, 2017
!------------------------------------------------------------------------------!
program euler2D
implicit none
include 'mpif.h'	!MPI header file

!Problem related variables
integer:: nx_global,ny_global
integer:: nx,ny,nt,iend,nsnap,iout,ifile,idt,nf,iprob,isgs,imodel,ifil,ix,nfil
integer:: iwriteI,iwriteF,npst
real*8 :: cfl,dt,dx,dy,lx,ly,time,tmax,eps,t1,t2,te,rate,te0,te_local,dt_local
real*8 :: gamma,a,b,dtout,tout,yp_min,ampx,ampy,sigma,re,pr,alpha,cfix,ukhi
real*8 :: denratio
real*8 :: kappa,capK,rth
integer:: i,j,m,n,ic,ie,iw,ny1,jip
real*8,allocatable:: q(:,:,:),u(:,:,:),s(:,:,:),x(:,:),y(:,:)

!MPI related variables
integer               :: myid
integer               :: np
integer               :: ierr
integer, dimension(48):: req = MPI_REQUEST_NULL
integer               :: status(MPI_STATUS_SIZE)
integer               :: status_array(MPI_STATUS_SIZE,48) !total 48 MPI resuests of send/receive
integer, parameter    :: id_top2bottom_1 = 1000 !message tag 
integer, parameter    :: id_bottom2top_1 = 1001 !message tag
integer, parameter    :: id_top2bottom_2 = 2000 !message tag 
integer, parameter    :: id_bottom2top_2 = 2001 !message tag
integer, parameter    :: id_top2bottom_3 = 3000 !message tag 
integer, parameter    :: id_bottom2top_3 = 3001 !message tag

common /fluids/ gamma
common /shock_trashold/ rth
common /instabilities/ lx,ly,ampx,ampy
common /weno_constant/ eps
common /problem/ iprob
common /modeling/ imodel,isgs
common /viscosity/ re,pr
common /filterstrength/ sigma
common /KHIalpha/ alpha,ukhi,denratio
common /fluxopt/ ix
common /entropyfix/ cfix,ie
common /waveopt/ iw !,ip,iq
common /PMconstant/ kappa,capK,npst

open(10,file='input.txt')
read(10,*)nx_global
read(10,*)cfl
read(10,*)tmax
read(10,*)iprob
read(10,*)nsnap
read(10,*)eps
read(10,*)alpha
read(10,*)denratio
read(10,*)ukhi
read(10,*)ampx
read(10,*)ampy
read(10,*)gamma
read(10,*)dt
read(10,*)idt
read(10,*)imodel
read(10,*)isgs
read(10,*)ifil
read(10,*)sigma
read(10,*)rth
read(10,*)nfil
read(10,*)re
read(10,*)pr
read(10,*)iwriteI
read(10,*)iwriteF
read(10,*)ix
read(10,*)cfix
read(10,*)ie
read(10,*)iw
read(10,*)kappa
read(10,*)capK
read(10,*)npst
close(10)

177 continue

!----------------------------------------------------------------------!
! MPI initialize
!----------------------------------------------------------------------!
      call MPI_INIT(ierr) 

      call MPI_Comm_size(MPI_COMM_WORLD, np, ierr) 

      call MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)
!----------------------------------------------------------------------!
!Global domain:
!----------------------------------------------------------------------!
!a unit square domain for KHI and SHOCK
ny_global = nx_global
lx = 1.0d0
ly = 1.0d0
if (iprob.eq.3) then !Redefine the domain for RTI (a rectangle with aspect ratio of 3)
ny_global = 3*nx_global
lx = 0.5d0
ly = 1.5d0
end if



!where the domain is defined in the interval of [-lx/2,+lx/2]x[-ly/2,+ly/2]
!with uniform spatial step size
dx = lx/dfloat(nx_global)
dy = ly/dfloat(ny_global)
      
!----------------------------------------------------------------------!
! Domain decomposition
! Overlapping grid for horizontal lines
!----------------------------------------------------------------------!

! Local array dimensions
  nx = nx_global
  !ny = ny_global/np
  ny1 = int(ny_global/np)
  jip = ny_global - np*ny1 - 1

    !load balancing
	if(myid.le.jip) then
    ny=ny1+1
    else
    ny=ny1
    end if 

    
	if (ny.lt.3) then !because of +3/-3 stencil
    nx_global = nx_global*2
    goto 177
    end if
    
! Local grid  
! cell-centered grid points:
allocate(x(-2:nx+3,-2:ny+3))
allocate(y(-2:nx+3,-2:ny+3))

!yp_min= -0.5d0*ly - 0.5d0*dy  + dfloat(myid)*dy*dfloat(ny)

if(myid.le.jip) then
yp_min= -0.5d0*ly - 0.5d0*dy  + dfloat(myid)*dy*dfloat(ny) 
else
yp_min= -0.5d0*ly - 0.5d0*dy  + dfloat(myid)*dy*dfloat(ny) + dfloat(jip+1)*dy  
end if

do j =-2, ny+3
do i =-2, nx+3
    x(i,j) = -0.5d0*lx - 0.5d0*dx + dfloat(i)*dx 
    y(i,j) = yp_min + dy*dfloat(j)
end do
end do



!----------------------------------------------------------------------!
!Problem definition and initialization
!iprob=1 ==> KHI problem
!iprob=2 ==> Riemann problem (Shock)
!iprob=3 ==> RTI problem 
!iprob=4 ==> RMI problem
!----------------------------------------------------------------------!
!use 3 gost cells within the domain

!Allocate local arrays
allocate(q(-2:nx+3,-2:ny+3,4)) !primary array for conservative field variables
allocate(u(-2:nx+3,-2:ny+3,4)) !temporary array for conservative field variables
allocate(s(nx,ny,4))           !rhs array

!initial conditions
if (iprob.eq.1) then
	call initializeKHI(nx,ny,x,y,q)
else if (iprob.eq.2) then
	call initializeSHOCK(nx,ny,x,y,q)
    if (tmax.ge.0.5d0) tmax = 0.5d0
else if (iprob.eq.3) then
    call initializeRTI(nx,ny,x,y,q)
    if (tmax.ge.4.5d0) tmax = 4.5d0
else !RMI
    call initializeRMI(nx,ny,x,y,q)      
end if



!counting info
iend = 0    
time = 0.0d0
dtout = tmax/dfloat(nsnap)
tout = dtout
iout = 0
ifile = 0
if (cfl.ge.1.0d0) cfl=1.0d0
if (nsnap .lt. 1) nsnap = 1 
   
if (idt.eq.1) then
nt = int(tmax/dt) !number of time step for constant dt
nf = nt/nsnap
else
nt = 1000000000 !some big number for maximum number of time step 
end if

!writing initial field (to make movie)
if (iwriteI.eq.1) then
call output(nx,ny,x,y,q,time,myid,ifile)
end if

!compute history (total energy = te )
call history(nx,ny,q,np,te_local)

	! Compute the total energy within all processors (MPI_SUM)
    call MPI_Reduce(te_local, te, 1, MPI_DOUBLE_PRECISION, &
                    MPI_SUM, 0, MPI_COMM_WORLD, ierr) 
 
	! Broadcast "te" so that each processor will have the same value 
    call MPI_Bcast(te, 1, MPI_DOUBLE_PRECISION, &
                   0,  MPI_COMM_WORLD, ierr)   

	te0 = te

if (myid.eq.0) then
open(17,file='rate.plt')
write(17,*) 'variables ="t","e"'

open(18,file='energy.plt')
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

	!determine time step to advance
	if (idt .eq. 1) then  !constant time step

    	if (mod(n,nf).eq.0) iout = 1
    
    else !compute adaptive time step
	
    	!Compute time step from cfl adaptively
		call timestep(nx,ny,dx,dy,cfl,dt_local,q)
   
		!Compute dt within all processors (MPI_MIN)
    	call MPI_Reduce(dt_local, dt, 1, MPI_DOUBLE_PRECISION, &
                    MPI_MIN, 0, MPI_COMM_WORLD, ierr) 
 
		!Broadcast "dt" so that each processor will have the same value 
    	call MPI_Bcast(dt, 1, MPI_DOUBLE_PRECISION, &
                   0,  MPI_COMM_WORLD, ierr)                  
    
    	!check for output times
    	if((time+dt) .ge. tout) then
    	dt = tout - time
    	tout=tout + dtout
    	iout=1
    	end if 

    	!check for final time step
    	if((time+dt) .ge. tmax) then
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
	call rhs(nx,ny,dx,dy,q,s)
    
    !update
	do m=1,4
    do j=1,ny
	do i=1,nx
    u(i,j,m) = q(i,j,m) + dt*s(i,j,m) 
    end do
    end do
    end do

    !update boundary conditions
    if (iprob.eq.1) then !periodic boundary conditions (global)

     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,4
        	ic = ic + 1
     		call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
        	call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    else if (iprob .eq. 2) then !open boundary conditions (global)
    	call openbc_bottom_top(nx,ny,u,myid,np)

    else if (iprob .eq. 3) then !reflective bc (global)  
        call refbc_bottom_top(nx,ny,u,myid,np)

	else !RMI - periodic left right and open bottom top

    	call openbc_bottom_top(nx,ny,u,myid,np)
  	        
    end if

  

    !SEND/RECEIVE DATA AMONG PROCESSORS
      
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
      	do m=1,4
        ic = ic + 1
     	call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
      	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
        call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do               
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)

if(nfil.eq.2) then    !Intermediate filtering step
	if (ifil.eq.2) then
		call filterSF7(nx,ny,u)
	else if (ifil.eq.3) then
		call filterTam7(nx,ny,u)
    else if (ifil.eq.4) then
      	call PMAD(nx,ny,u,dt,dx)
    else if (ifil.eq.5) then
      	call shockfilterSF7(nx,ny,u)
	end if

		!Exchange filtered data among processors
    if (ifil.eq.2 .or. ifil.eq.3 .or. ifil.eq.4 .or. ifil.eq.5) then
    	!update boundary conditions
    	if (iprob.eq.1) then !periodic boundary conditions (global)

     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,4
        	ic = ic + 1
     		call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
        	call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    	else if (iprob .eq. 2) then !open boundary conditions (global)
    		call openbc_bottom_top(nx,ny,u,myid,np)

    	else if (iprob .eq. 3) then !reflective bc (global)  
        	call refbc_bottom_top(nx,ny,u,myid,np)

		else !RMI  periodic left right and open bottom top          

    		call openbc_bottom_top(nx,ny,u,myid,np)
         
    	end if
    
    !SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
       	do m=1,4
        ic = ic + 1
     	call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
        call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do                
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)
    
    end if !End of filtered data exchange





    
end if !End of intermediate filtering

    
    !--------------------!
	!Step 2
    !--------------------!
	call rhs(nx,ny,dx,dy,u,s)

	!update
	do m=1,4
    do j=1,ny
	do i=1,nx
    u(i,j,m) = 0.75d0*q(i,j,m) + 0.25d0*u(i,j,m) + 0.25d0*dt*s(i,j,m) 
    end do
    end do
    end do

    !update boundary conditions
    if (iprob.eq.1) then

     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then !periodic boundary conditions (global)
       
       		do m=1,4
        	ic = ic + 1
     		call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
        	call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    else if (iprob .eq. 2) then !open boundary conditions (global)
    	call openbc_bottom_top(nx,ny,u,myid,np)

    else if (iprob .eq. 3) then !reflective bc (global)  !RTI
        call refbc_bottom_top(nx,ny,u,myid,np)

	else !RMI  periodic left right and open bottom top          

    	call openbc_bottom_top(nx,ny,u,myid,np)
        
    end if
    
   	!SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
       	do m=1,4
        ic = ic + 1
     	call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
        call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
        end do               
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)

if(nfil.eq.2) then    
	if (ifil.eq.2) then
		call filterSF7(nx,ny,u)
	else if (ifil.eq.3) then
		call filterTam7(nx,ny,u)
    else if (ifil.eq.4) then
      	call PMAD(nx,ny,u,dt,dx)
    else if (ifil.eq.5) then
      	call shockfilterSF7(nx,ny,u)
	end if

		!Exchange filtered data among processors
    if (ifil.eq.2 .or. ifil.eq.3 .or. ifil.eq.4 .or. ifil.eq.5) then
    	!update boundary conditions
    	if (iprob.eq.1) then !periodic boundary conditions (global)

     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,4
        	ic = ic + 1
     		call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
        	call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    	else if (iprob .eq. 2) then !open boundary conditions (global)
    		call openbc_bottom_top(nx,ny,q,myid,np)

    	else if (iprob .eq. 3) then !reflective bc (global)  
        	call refbc_bottom_top(nx,ny,q,myid,np)
		
        else!RMI  periodic left right and open bottom top          

    		call openbc_bottom_top(nx,ny,u,myid,np)        
    	
	end if
    
    !SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
       	do m=1,4
        ic = ic + 1
     	call MPI_Isend(u(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(u(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(u(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Isend(u(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(u(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
        call MPI_Irecv(u(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(u(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(u(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do                
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)
    
    end if !end of filtered data exchange
    
end if	!end of intermediate filtering    

    
    !--------------------!
	!Step 3 
    !--------------------! 
	call rhs(nx,ny,dx,dy,u,s)
    
	!update
	do m=1,4
    do j=1,ny
	do i=1,nx
    q(i,j,m) = a*q(i,j,m)+b*u(i,j,m)+b*dt*s(i,j,m) 
    end do
    end do 
    end do
   
    !update boundary conditions
    if (iprob.eq.1) then !periodic boundary conditions (global)

     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,4
        	ic = ic + 1
     		call MPI_Isend(q(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(q(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(q(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Isend(q(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
        	call MPI_Irecv(q(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Irecv(q(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    else if (iprob .eq. 2) then !open boundary conditions (global)
    	call openbc_bottom_top(nx,ny,q,myid,np)

    else if (iprob .eq. 3) then !reflective bc (global)  
        call refbc_bottom_top(nx,ny,q,myid,np)

	else		!RMI  periodic left right and open bottom top          

    	call openbc_bottom_top(nx,ny,q,myid,np)

        
    end if
    
    !SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
       	do m=1,4
        ic = ic + 1
     	call MPI_Isend(q(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(q(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(q(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Isend(q(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(q(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(q(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
        call MPI_Irecv(q(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(q(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(q(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Irecv(q(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(q(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(q(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do                
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)


	!----------------------------------------------------------------------!
    !Filtering (at the end of each time step)
    !----------------------------------------------------------------------!
	if (ifil.eq.2) then
		call filterSF7(nx,ny,q)
	else if (ifil.eq.3) then
		call filterTam7(nx,ny,q)
    else if (ifil.eq.4) then
      	call PMAD(nx,ny,q,dt,dx)
    else if (ifil.eq.5) then
      	call shockfilterSF7(nx,ny,q)        
	end if 
    
	!Exchance filtered data among processors
    if (ifil.eq.2 .or. ifil.eq.3 .or. ifil.eq.4 .or. ifil.eq.5) then
    	!update boundary conditions
    	if (iprob.eq.1) then !periodic boundary conditions (global)

     	ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then
       
       		do m=1,4
        	ic = ic + 1
     		call MPI_Isend(q(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Isend(q(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        	ic = ic + 1
        	call MPI_Isend(q(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Isend(q(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Isend(q(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
       		do m=1,4
        	ic = ic + 1
        	call MPI_Irecv(q(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
        	call MPI_Irecv(q(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
       		do m=1,4
        	ic = ic + 1
       		call MPI_Irecv(q(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        	ic = ic + 1
       		call MPI_Irecv(q(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       		end do               
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    	else if (iprob .eq. 2) then !open boundary conditions (global)
    		call openbc_bottom_top(nx,ny,q,myid,np)

    	else if (iprob .eq. 3) then !reflective bc (global)  
        	call refbc_bottom_top(nx,ny,q,myid,np)


		else !RMI  periodic left right and open bottom top          

    		call openbc_bottom_top(nx,ny,q,myid,np)
    	end if
    
    !SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
       
       	do m=1,4
        ic = ic + 1
     	call MPI_Isend(q(1,ny,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Isend(q(1,ny-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)

        ic = ic + 1
        call MPI_Isend(q(1,ny-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Isend(q(1,1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(q(1,2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Isend(q(1,3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
       	do m=1,4
        ic = ic + 1
        call MPI_Irecv(q(1,0,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(q(1,-1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
        call MPI_Irecv(q(1,-2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_3, MPI_COMM_WORLD, req(ic), ierr)
       end do
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
       	do m=1,4
        ic = ic + 1
       	call MPI_Irecv(q(1,ny+1,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(q(1,ny+2,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_2, MPI_COMM_WORLD, req(ic), ierr)
        ic = ic + 1
       	call MPI_Irecv(q(1,ny+3,m), nx, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_3, MPI_COMM_WORLD, req(ic), ierr)
       	end do                
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)
    
    end if !end of filtering

    !----------------------------------------------------------------------!               
	!log file
    !----------------------------------------------------------------------!
    if (myid.eq.0) write(*,*)n,time

    !----------------------------------------------------------------------!
    !I/O Files
    !----------------------------------------------------------------------!
    if (iout.eq.1.and.iwriteF.eq.1) then  
      
    ifile = ifile + 1

    !----------------------------------------------------------------------!
	! Wait until all processors come to this point before writing files
	!----------------------------------------------------------------------!
	call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    !write output for all processors
	call output(nx,ny,x,y,q,time,myid,ifile)
    
    iout = 0
    end if

    !----------------------------------------------------------------------!
	!Post-processing:
	!----------------------------------------------------------------------!
    
	!compute history (total energy = te)
	call history(nx,ny,q,np,te_local)

    ! Compute the total energy within all processors (MPI_SUM)
    call MPI_Reduce(te_local, te, 1, MPI_DOUBLE_PRECISION, &
                    MPI_SUM, 0, MPI_COMM_WORLD, ierr) 
 
	! Broadcast "te" so that each processor will have the same value 
    call MPI_Bcast(te, 1, MPI_DOUBLE_PRECISION, &
                   0,  MPI_COMM_WORLD, ierr)

                   
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

if (myid.eq.0) close(17)
if (myid.eq.0) close(18)
     
!write CPU data
if (myid.eq.0) then
open(19,file='cpu.txt') 
write(19,*)"number of processor =", np
write(19,*)""
write(19,*)"local resolution    =", nx,ny
write(19,*)"global resolution   =", nx_global,ny_global
write(19,*)""
write(19,*)"cpu time =", t2-t1, "  seconds"
write(19,*)"cpu time =", (t2-t1)/60.0d0, "  minutes"
write(19,*)"cpu time =", (t2-t1)/3600.0d0, "  hours" 
write(19,*)""
write(19,*)"time step =", dt
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
! Output files
!----------------------------------------------------------------------!
subroutine output(nx,ny,x,y,q,time,myid,ifile)
implicit none
integer::nx,ny,myid,ifile
real*8 ::time,p,gamma
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8 ::x(-2:nx+3,-2:ny+3)
real*8 ::y(-2:nx+3,-2:ny+3)
integer:: i,j
character(80):: charID, snapID, timeID, filename

common /fluids/ gamma

write(charID,'(i5)') myid       !index for each processor 
write(snapID,'(i5)') ifile      !index for time snapshot
write(timeID,'(f12.4)') time    !index for plotting time

! Define the file name
filename = "density_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.plt'

! Open the file and start writing the data
open(unit=19, file=filename)

! Tecplot header
write(19,*) 'title =', '"Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID))// &
            '_Time_'// trim(adjustl(timeID)), '"'
write(19,*) 'variables = "x", "y", "Density"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID)), &
            ' i=', nx+1, ' j=', ny+1, ' f=point'

! Write density data
do j = 0, ny
do i = 0, nx
  	write(19,*) x(i,j), y(i,j), q(i,j,1)
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
write(19,*) 'variables = "x", "y", "p"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID)), &
            ' i=', nx+1, ' j=', ny+1, ' f=point'

! Write density data
do j = 0, ny
do i = 0, nx
p = (gamma-1.0d0)*(q(i,j,4)-0.5d0*(q(i,j,2)*q(i,j,2)/q(i,j,1) &
    +q(i,j,3)*q(i,j,3)/q(i,j,1)))
write(19,*) x(i,j), y(i,j), p
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
write(19,*) 'variables = "x", "y", "u","v"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID)), &
            ' i=', nx+1, ' j=', ny+1, ' f=point'

! Write velocity data
do j = 0, ny
do i = 0, nx
  	write(19,*) x(i,j), y(i,j), q(i,j,2)/q(i,j,1), q(i,j,3)/q(i,j,1)
end do 
end do

close(19)


! Write load information
! Define the file name
filename = "load_"// trim(adjustl(charID)) // '.dat'
open(unit=19, file=filename)
write(19,*)myid,ny
close(19)

end


!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!KHI: Kelvin-Helmholtz Instability
!-----------------------------------------------------------------------------------!
subroutine initializeKHI(nx,ny,x,y,q)
implicit none
integer::nx,ny,i,j
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8 ::x(-2:nx+3,-2:ny+3)
real*8 ::y(-2:nx+3,-2:ny+3)
real*8 ::r,u,v,p,e,ukhi
real*8 ::gamma,lx,ly,ampx,ampy,pi,alpha,denratio

common /fluids/ gamma
common /instabilities/ lx,ly,ampx,ampy
common /KHIalpha/ alpha,ukhi,denratio

!alpha=1.98295d0 !Old version value

pi =4.0d0*datan(1.0d0)

do j=-2,ny+3
do i=-2,nx+3 
   
  	if(dabs(y(i,j)).ge.0.25d0) then !outer region     
    	r = 1.0d0
		u = ukhi + ampx*dsin(2.0d0*pi*alpha*x(i,j)/lx)
    	v = 0.0d0 + ampy*dsin(2.0d0*pi*alpha*x(i,j)/lx)
    	p = 2.5d0          
    else !inner region
        r = denratio
		u =-ukhi - ampx*dsin(2.0d0*pi*alpha*x(i,j)/lx)
    	v = 0.0d0 + ampy*dsin(2.0d0*pi*alpha*x(i,j)/lx)
    	p = 2.5d0       
    end if
    	
		e=p/(r*(gamma-1.0d0))+0.5d0*(u*u+v*v)
        
    !construct conservative variables     
	q(i,j,1)=r
	q(i,j,2)=r*u
    q(i,j,3)=r*v
	q(i,j,4)=r*e    
end do
end do

call perbc_left_right(nx,ny,q)


return
end



!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!SHOCK (Lax-Lio Configuration 3)
!-----------------------------------------------------------------------------------!
subroutine initializeSHOCK(nx,ny,x,y,q)
implicit none
integer::nx,ny,i,j
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8 ::x(-2:nx+3,-2:ny+3)
real*8 ::y(-2:nx+3,-2:ny+3)
real*8 ::r,u,v,p,e
real*8 ::gamma
real*8 ::rUL,rUR,rLL,rLR,uUL,uUR,uLL,uLR,vUL,vUR,vLL,vLR,pUL,pUR,pLL,pLR

common /fluids/ gamma

  	!upper-left
    rUL=0.5323d0
	uUL=1.206d0
    vUL=0.0d0
    pUL=0.3d0	
	!upper-right
    rUR=1.5d0
	uUR=0.0d0
    vUR=0.0d0
    pUR=1.5d0	
	!lower-left
    rLL=0.138d0
	uLL=1.206d0
    vLL=1.206d0
    pLL=0.029d0	
	!lower-right
    rLR=0.5323d0
	uLR=0.0d0
    vLR=1.206d0
    pLR=0.3d0	
    
do j=-2,ny+3
do i=-2,nx+3
  
    
  	if(x(i,j) .le. 0.0d0 .and. y(i,j) .gt. 0.0d0) then !upper-left
    
    
    	r = rUL
		u = uUL
    	v = vUL
    	p = pUL
        
	else if(x(i,j) .gt. 0.0d0 .and. y(i,j) .gt. 0.0d0) then !upper-right
    
    
       	r = rUR
		u = uUR
    	v = vUR
    	p = pUR
   
	else if(x(i,j) .gt. 0.0d0 .and. y(i,j) .le. 0.0d0) then !lower-right
        
		r = rLR
		u = uLR
    	v = vLR
    	p = pLR  
   
    else !lower-left

        r = rLL
		u = uLL
    	v = vLL
    	p = pLL
        
        
    end if
    	
		e=p/(r*(gamma-1.0d0))+0.5d0*(u*u+v*v)
    
    
    !construct conservative variables 
    
	q(i,j,1)=r
	q(i,j,2)=r*u
    q(i,j,3)=r*v
	q(i,j,4)=r*e
    
end do
end do


return
end


!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!RTI: Rayleigh-Taylor Instability
!-----------------------------------------------------------------------------------!
subroutine initializeRTI(nx,ny,x,y,q)
implicit none
integer::nx,ny,i,j
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8 ::x(-2:nx+3,-2:ny+3)
real*8 ::y(-2:nx+3,-2:ny+3)
real*8 ::r,u,v,p,e
real*8 ::gamma,lx,ly,ampx,ampy,pi

common /fluids/ gamma
common /instabilities/ lx,ly,ampx,ampy



pi = 4.0d0*datan(1.0d0)

do j=-2,ny+3
do i=-2,nx+3 
   
  	if (y(i,j) .ge. 0.0d0) then !upper
     
    	r = 2.0d0
		u = 0.0d0
    	p = 2.5d0 - r*y(i,j)
        
    else !lower

        r = 1.0d0
		u = 0.0d0
    	p = 2.5d0 - r*y(i,j)
        
    end if
	        
	!perturbation
	v = (ampy/4.0d0)*(1.0d0 + dcos(2.0d0*pi*x(i,j)/lx))*(1.0d0 + dcos(2.0d0*pi*y(i,j)/ly))
  
    e=p/(r*(gamma-1.0d0))+0.5d0*(u*u+v*v) 
   
    !construct conservative variables    
	q(i,j,1)=r
	q(i,j,2)=r*u
    q(i,j,3)=r*v
	q(i,j,4)=r*e
    
end do
end do


return
end

!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!RMI: Richtmeyer-Meshkov Instability
!-----------------------------------------------------------------------------------!
subroutine initializeRMI(nx,ny,x,y,q)
implicit none
integer::nx,ny,i,j
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8 ::x(-2:nx+3,-2:ny+3)
real*8 ::y(-2:nx+3,-2:ny+3)
real*8 ::lbd,mbd
real*8 ::r,u,v,p,e,gamma
real*8 ::pi

common /fluids/ gamma


pi = 4.0d0*datan(1.0d0)
lbd = 0.06d0

do i = -2,nx+3
  do j = -2,ny+3

    mbd = 0.1d0+0.008d0*dcos(20.0d0*pi*x(i,j))

    if (y(i,j)<=lbd) then
      q(i,j,1) = 2.67d0
      q(i,j,2) = q(i,j,1)*0.0d0
      q(i,j,3) = q(i,j,1)*1.48d0

      p = 4.5d0
      e=p/(q(i,j,1)*(gamma-1.0d0))+0.5d0*(1.48d0*1.48d0)
      q(i,j,4) = q(i,j,1)*e

    else if (y(i,j)>lbd.and.y(i,j)<mbd) then

      q(i,j,1) = 1.0d0
      q(i,j,2) = q(i,j,1)*0.0d0
      q(i,j,3) = q(i,j,1)*0.0d0

      p = 1.0d0
      e=p/(q(i,j,1)*(gamma-1.0d0))+0.5d0*(0.0d0*0.0d0)
      q(i,j,4) = q(i,j,1)*e      

	else if (y(i,j)>mbd) then

      q(i,j,1) = 0.138d0
      q(i,j,2) = q(i,j,1)*0.0d0
      q(i,j,3) = q(i,j,1)*0.0d0

      p = 1.0d0
      e=p/(q(i,j,1)*(gamma-1.0d0))+0.5d0*(0.0d0*0.0d0)
      q(i,j,4) = q(i,j,1)*e    
  
	end if

end do
end do  

call perbc_left_right(nx,ny,q)

return
end


!-----------------------------------------------------------------------------------!
!Transmissive boundary conditions (open bc) in y-direction
!-----------------------------------------------------------------------------------!
subroutine openbc_bottom_top(nx,ny,q,myid,np)
implicit none
integer::nx,ny,i,j,m,myid,np
real*8 ::q(-2:nx+3,-2:ny+3,4)


if (myid .eq. 0) then 
    do m=1,4
	do i=1,nx
	q(i, 0,m)  =q(i,1,m) 	!bottom of the domain (j= 0)
    q(i,-1,m)  =q(i,2,m)	!bottom of the domain (j=-1)
    q(i,-2,m)  =q(i,3,m)	!bottom of the domain (j=-2)
    end do
    end do
end if
    

if (myid .eq. np-1) then
    do m=1,4
	do i=1,nx
	q(i,ny+1,m)=q(i,ny,m) 	!top of the domain	(j=ny+1)
	q(i,ny+2,m)=q(i,ny-1,m) !top of the domain	(j=ny+2)
	q(i,ny+3,m)=q(i,ny-2,m) !top of the domain  (j=my+3)
	end do
    end do
end if
     


return
end


!-----------------------------------------------------------------------------------!
!Reflective boundary conditions (open bc) in y-direction
!-----------------------------------------------------------------------------------!
subroutine refbc_bottom_top(nx,ny,q,myid,np)
implicit none
integer::nx,ny,i,j,m,myid,np
real*8 ::q(-2:nx+3,-2:ny+3,4)


if (myid .eq. 0) then 
    do m=1,4
    if (m .eq. 2) cycle 
	if (m .eq. 3) cycle 
	do i=1,nx
	q(i, 0,m)  =q(i,1,m) 	!bottom of the domain (j= 0)
    q(i,-1,m)  =q(i,2,m)	!bottom of the domain (j=-1)
    q(i,-2,m)  =q(i,3,m)	!bottom of the domain (j=-2)
    end do
    end do

    !b.c. (u=0)
	m=2 
	do i=1,nx
	q(i, 0,m)  =-q(i,1,m) 	!bottom of the domain
    q(i,-1,m)  =-q(i,2,m)	!bottom of the domain
    q(i,-2,m)  =-q(i,3,m)	!bottom of the domain
	end do

    !b.c. (v=0)
	m=3 
	do i=1,nx
	q(i, 0,m)  =-q(i,1,m) 	!bottom of the domain
    q(i,-1,m)  =-q(i,2,m)	!bottom of the domain
    q(i,-2,m)  =-q(i,3,m)	!bottom of the domain
	end do
    
end if
    

if (myid .eq. np-1) then
    do m=1,4
    if (m .eq. 2) cycle 
	if (m .eq. 3) cycle 
	do i=1,nx
	q(i,ny+1,m)=q(i,ny,m) 	!top of the domain	(j=ny+1)
	q(i,ny+2,m)=q(i,ny-1,m) !top of the domain	(j=ny+2)
	q(i,ny+3,m)=q(i,ny-2,m) !top of the domain  (j=my+3)
	end do
    end do

    !b.c. (u=0)
	m=2 
	do i=1,nx 
	q(i,ny+1,m)=-q(i,ny,m) 	 !top of the domain	
	q(i,ny+2,m)=-q(i,ny-1,m) !top of the domain	
	q(i,ny+3,m)=-q(i,ny-2,m) !top of the domain
	end do

    !b.c. (v=0)
	m=3 
	do i=1,nx  
	q(i,ny+1,m)=-q(i,ny,m) 	 !top of the domain	
	q(i,ny+2,m)=-q(i,ny-1,m) !top of the domain	
	q(i,ny+3,m)=-q(i,ny-2,m) !top of the domain
	end do
    
end if
     


return
end


!-----------------------------------------------------------------------------------!
!Transmissive boundary conditions (open bc) in x-direction
!-----------------------------------------------------------------------------------!
subroutine openbc_left_right(nx,ny,q)
implicit none
integer::nx,ny,i,j,m
real*8 ::q(-2:nx+3,-2:ny+3,4)

do m=1,4
  
	do j=-2,ny+3
	q( 0,j,m)  =q(1,j,m)	!left of the domain
    q(-1,j,m)  =q(2,j,m)	!left of the domain
    q(-2,j,m)  =q(3,j,m)	!left of the domain
  
	q(nx+1,j,m)=q(nx,j,m) 	!right of the domain	
	q(nx+2,j,m)=q(nx-1,j,m)	!right of the domain	
	q(nx+3,j,m)=q(nx-2,j,m)	!right of the domain
	end do
    
end do

return
end


!-----------------------------------------------------------------------------------!
!Periodic boundary conditions in x-direction
!-----------------------------------------------------------------------------------!
subroutine perbc_left_right(nx,ny,q)
implicit none
integer::nx,ny,i,j,m
real*8 ::q(-2:nx+3,-2:ny+3,4)


do m=1,4
   
	do j=-2,ny+3
	q( 0,j,m)  =q(nx,j,m)	!left of the domain
    q(-1,j,m)  =q(nx-1,j,m)	!left of the domain
    q(-2,j,m)  =q(nx-2,j,m)	!left of the domain
    
	q(nx+1,j,m)=q(1,j,m) 	!right of the domain	
	q(nx+2,j,m)=q(2,j,m)	!right of the domain	
	q(nx+3,j,m)=q(3,j,m)	!right of the domain
	end do
    
end do

return
end


!-----------------------------------------------------------------------------------!
!Boundary Conditions
!-----------------------------------------------------------------------------------!
subroutine bcs(nx,ny,q)
implicit none
integer::nx,ny,iprob
real*8 ::q(-2:nx+3,-2:ny+3,4)

common /problem/ iprob


!Update BC
!--------------------------------------------!
!apply boundary condition for left and right 
!--------------------------------------------!
if(iprob.eq.1) then !KHI
  
	!-------------------------------------------------!
	!Periodic boundary condition
	!-------------------------------------------------!
	call perbc_left_right(nx,ny,q)

else if(iprob.eq.2) then !SHOCK

	!-------------------------------------------------!
	!Open boundary condition
	!-------------------------------------------------!
	call openbc_left_right(nx,ny,q)

else if(iprob.eq.3) then!RTI

  	!-------------------------------------------------!
	!Periodic boundary condition
	!-------------------------------------------------!
	call perbc_left_right(nx,ny,q)

else     !RMI

  	!-------------------------------------------------!
	!Open boundary condition
	!-------------------------------------------------!
	call perbc_left_right(nx,ny,q)

end if

      

return
end
      
!-----------------------------------------------------------------------------------!
!Filtering (SF-7)
!-----------------------------------------------------------------------------------!
subroutine filterSF7(nx,ny,q)
implicit none
integer::nx,ny,i,j,m
real*8 ::sigma,d0,d1,d2,d3
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8,allocatable:: u(:,:,:),v(:,:,:)

common /filterstrength/ sigma

!Update BC
call bcs(nx,ny,q)

allocate(u(-2:nx+3,-2:ny+3,4))
allocate(v(-2:nx+3,-2:ny+3,4))

do m=1,4
do j=-2,ny+3
do i=-2,nx+3
u(i,j,m) = q(i,j,m)
end do
end do
end do


d0 = 5.0d0/16.0d0
d1 =-15.0d0/64.0d0
d2 = 3.0d0/32.0d0
d3 =-1.0d0/64.0d0

!do m=1,4
!do j=1,ny
!do i=1,nx
!q(i,j,m) = u(i,j,m) - sigma*(2.0d0*d0*u(i,j,m) &
                    !+ d1*(u(i+1,j,m)+u(i-1,j,m)+u(i,j+1,m)+u(i,j-1,m)) &
                    !+ d2*(u(i+2,j,m)+u(i-2,j,m)+u(i,j+2,m)+u(i,j-2,m)) &
                    !+ d3*(u(i+3,j,m)+u(i-3,j,m)+u(i,j+3,m)+u(i,j-3,m)) )                               
!end do
!end do
!end do

!filter in y
do m=1,4
do j=1,ny
do i=1,nx
v(i,j,m) = u(i,j,m) - sigma*(d0*u(i,j,m) &
                           + d1*(u(i,j+1,m)+u(i,j-1,m)) &
                           + d2*(u(i,j+2,m)+u(i,j-2,m)) &
                           + d3*(u(i,j+3,m)+u(i,j-3,m)) )                               
end do
end do
end do


call bcs(nx,ny,v)

!filter in x
do m=1,4
do j=1,ny
do i=1,nx
q(i,j,m) = v(i,j,m) - sigma*(d0*v(i,j,m) &
                           + d1*(v(i+1,j,m)+v(i-1,j,m)) &
                           + d2*(v(i+2,j,m)+v(i-2,j,m)) &
                           + d3*(v(i+3,j,m)+v(i-3,j,m)) )                               
end do
end do
end do

call bcs(nx,ny,q)



deallocate(u,v)

return
end


!-----------------------------------------------------------------------------------!
!Filtering (Tam-7)
!-----------------------------------------------------------------------------------!
subroutine filterTam7(nx,ny,q)
implicit none
integer::nx,ny,i,j,m
real*8 ::sigma,d0,d1,d2,d3
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8,allocatable:: u(:,:,:),v(:,:,:)

common /filterstrength/ sigma

!Update BC
call bcs(nx,ny,q)


allocate(u(-2:nx+3,-2:ny+3,4))
allocate(v(-2:nx+3,-2:ny+3,4))

do m=1,4
do j=-2,ny+3
do i=-2,nx+3
u(i,j,m) = q(i,j,m)
end do
end do
end do


d0 = 0.287392842460d0
d1 =-0.226146951809d0
d2 = 0.106303578770d0
d3 =-0.023853048191d0

!do m=1,4
!do j=1,ny
!do i=1,nx
!q(i,j,m) = u(i,j,m) - sigma*(2.0d0*d0*u(i,j,m) &
                    !+ d1*(u(i+1,j,m)+u(i-1,j,m)+u(i,j+1,m)+u(i,j-1,m)) &
                    !+ d2*(u(i+2,j,m)+u(i-2,j,m)+u(i,j+2,m)+u(i,j-2,m)) &
                    !+ d3*(u(i+3,j,m)+u(i-3,j,m)+u(i,j+3,m)+u(i,j-3,m)) )                             
!end do
!end do
!end do

!filter in y
do m=1,4
do j=1,ny
do i=1,nx
v(i,j,m) = u(i,j,m) - sigma*(d0*u(i,j,m) &
                           + d1*(u(i,j+1,m)+u(i,j-1,m)) &
                           + d2*(u(i,j+2,m)+u(i,j-2,m)) &
                           + d3*(u(i,j+3,m)+u(i,j-3,m)) )                               
end do
end do
end do


call bcs(nx,ny,v)

!filter in x
do m=1,4
do j=1,ny
do i=1,nx
q(i,j,m) = v(i,j,m) - sigma*(d0*v(i,j,m) &
                           + d1*(v(i+1,j,m)+v(i-1,j,m)) &
                           + d2*(v(i+2,j,m)+v(i-2,j,m)) &
                           + d3*(v(i+3,j,m)+v(i-3,j,m)) )                               
end do
end do
end do

call bcs(nx,ny,q)



deallocate(u,v)


return
end

!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS)
!-----------------------------------------------------------------------------------!
subroutine rhs(nx,ny,dx,dy,q,s)
implicit none
integer::nx,ny,imodel,isgs
real*8 ::dx,dy
real*8 ::q(-2:nx+3,-2:ny+3,4),s(nx,ny,4)

common /modeling/ imodel,isgs


!Update BC
call bcs(nx,ny,q)

!Compute RHS
 
if (imodel.eq.1) then !Euler Model
    if (isgs.eq.1) then !WENO for inviscid term
	call rhsILES(nx,ny,dx,dy,q,s)
    else !Central for inviscid term
    call rhsCS(nx,ny,dx,dy,q,s)  
    end if
else !NS Model
  	!Viscous terms are always central difference
	if (isgs.eq.1) then !WENO for inviscid term
    call rhsILES(nx,ny,dx,dy,q,s)
    call rhsVIS(nx,ny,dx,dy,q,s)
    else !Central for inviscid term
    call rhsCS(nx,ny,dx,dy,q,s)
    call rhsVIS(nx,ny,dx,dy,q,s)	
    end if
end if
  
return 
end 

!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS) for viscous term (central)
!-----------------------------------------------------------------------------------!
subroutine rhsVIS(nx,ny,dx,dy,q,s)
implicit none
integer::nx,ny,i,j,m
real*8 ::dx,dy,re,pr,gamma
real*8 ::g,a,b,c,dx1,dx3,dx5,dy1,dy3,dy5
real*8 ::g1,g2,g3,u,v,txx,tyy,txy,tx,ty,ux,uy,vx,vy,qx,qy
real*8 ::q(-2:nx+3,-2:ny+3,4),s(nx,ny,4)
real*8,allocatable:: t(:,:),vf(:,:,:),vg(:,:,:)

common /viscosity/ re,pr
common /fluids/ gamma


allocate(t(-2:nx+3,-2:ny+3))
allocate(vf(0:nx,0:ny,4),vg(0:nx,0:ny,4))

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
g3 =-gamma/(re*pr)


dx1 = 1.0d0/dx
dx3 = 1.0d0/(3.0d0*dx)
dx5 = 1.0d0/(5.0d0*dx)

dy1 = 1.0d0/dy
dy3 = 1.0d0/(3.0d0*dy)
dy5 = 1.0d0/(5.0d0*dy)

!compute temperature
do j=-2,ny+3
do i=-2,nx+3
t(i,j) = q(i,j,4)/q(i,j,1) &
       - 0.5d0*((q(i,j,2)/q(i,j,1))*(q(i,j,2)/q(i,j,1)) &
               +(q(i,j,3)/q(i,j,1))*(q(i,j,3)/q(i,j,1)) )
end do
end do

!compute viscous fluxes at cell interfaces:

do j=0,ny
do i=0,nx

ux = g*(  a*dx1*( q(i+1,j,2)/q(i+1,j,1) - q(i,j,2)/q(i,j,1) ) &
        + b*dx3*( q(i+2,j,2)/q(i+2,j,1) - q(i-1,j,2)/q(i-1,j,1) ) &
        + c*dx5*( q(i+3,j,2)/q(i+3,j,1) - q(i-2,j,2)/q(i-2,j,1) ) )

uy = g*(a*dy1*( q(i,j+1,2)/q(i,j+1,1) - q(i,j,2)/q(i,j,1) ) &
      + b*dy3*( q(i,j+2,2)/q(i,j+2,1) - q(i,j-1,2)/q(i,j-1,1) ) &
      + c*dy5*( q(i,j+3,2)/q(i,j+3,1) - q(i,j-2,2)/q(i,j-2,1) ) )

vx = g*(a*dx1*( q(i+1,j,3)/q(i+1,j,1) - q(i,j,3)/q(i,j,1) ) &
      + b*dx3*( q(i+2,j,3)/q(i+2,j,1) - q(i-1,j,3)/q(i-1,j,1) ) &
      + c*dx5*( q(i+3,j,3)/q(i+3,j,1) - q(i-2,j,3)/q(i-2,j,1) ) )

vy = g*(a*dy1*( q(i,j+1,3)/q(i,j+1,1) - q(i,j,3)/q(i,j,1) ) &
      + b*dy3*( q(i,j+2,3)/q(i,j+2,1) - q(i,j-1,3)/q(i,j-1,1) ) &
      + c*dy5*( q(i,j+3,3)/q(i,j+3,1) - q(i,j-2,3)/q(i,j-2,1) ) )
      

tx = g*(a*dx1*( t(i+1,j) - t(i,j) ) &
      + b*dx3*( t(i+2,j) - t(i-1,j) ) &
      + c*dx5*( t(i+3,j) - t(i-2,j) ) )

ty = g*(a*dy1*( t(i,j+1) - t(i,j) ) &
      + b*dy3*( t(i,j+2) - t(i,j-1) ) &
      + c*dy5*( t(i,j+3) - t(i,j-2) ) )

txx = g2*(2.0d0*ux - vy) 
tyy = g2*(2.0d0*vy - ux) 
txy = g1*(uy + vx) 
qx  = g3*tx
qy  = g3*ty         
u = q(i,j,2)/q(i,j,1)
v = q(i,j,3)/q(i,j,1)

vf(i,j,1) = 0.0d0
vf(i,j,2) = txx
vf(i,j,3) = txy
vf(i,j,4) = u*txx + v*txy - qx

vg(i,j,1) = 0.0d0
vg(i,j,2) = txy
vg(i,j,3) = tyy
vg(i,j,4) = u*txy + v*tyy - qy 

end do
end do

! compute RHS contribution due to viscous term (central difference)
do m=1,4
do j=1,ny
do i=1,nx

s(i,j,m) = s(i,j,m) + (vf(i,j,m)-vf(i-1,j,m))/dx &
                    + (vg(i,j,m)-vg(i,j-1,m))/dy 

end do
end do
end do

deallocate(t,vf,vg)

return
end

!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS) for inviscid term (central)
!-----------------------------------------------------------------------------------!
subroutine rhsCS(nx,ny,dx,dy,q,s)
implicit none
integer::nx,ny,i,j,m,iprob
real*8 ::dx,dy,gamma
real*8 ::g,a,b,c,gm,uu,vv,pp,rh,re
real*8 ::q(-2:nx+3,-2:ny+3,4),s(nx,ny,4)
real*8,allocatable:: u(:,:),v(:,:),p(:,:),vf(:,:,:),vg(:,:,:)

common /fluids/ gamma
common /problem/ iprob

allocate(u(-2:nx+3,-2:ny+3),v(-2:nx+3,-2:ny+3),p(-2:nx+3,-2:ny+3))
allocate(vf(0:nx,0:ny,4),vg(0:nx,0:ny,4))

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
do j=-2,ny+3
do i=-2,nx+3
u(i,j) = q(i,j,2)/q(i,j,1)
v(i,j) = q(i,j,3)/q(i,j,1)
p(i,j) = gm*(q(i,j,4)- 0.5d0*(q(i,j,2)*u(i,j)+q(i,j,3)*v(i,j)))
end do
end do

!compute fluxes at cell interfaces:x

do j=0,ny
do i=0,nx

uu = g*(a*( u(i+1,j) + u(i,j) ) &
      + b*( u(i+2,j) + u(i-1,j) ) &
      + c*( u(i+3,j) + u(i-2,j) ) )
      
vv = g*(a*( v(i+1,j) + v(i,j) ) &
      + b*( v(i+2,j) + v(i-1,j) ) &
      + c*( v(i+3,j) + v(i-2,j) ) )
      
pp = g*(a*( p(i+1,j) + p(i,j) ) &
      + b*( p(i+2,j) + p(i-1,j) ) &
      + c*( p(i+3,j) + p(i-2,j) ) )

rh = g*(a*( q(i+1,j,1) + q(i,j,1) ) &
      + b*( q(i+2,j,1) + q(i-1,j,1) ) &
      + c*( q(i+3,j,1) + q(i-2,j,1) ) )

re = g*(a*( q(i+1,j,4) + q(i,j,4) ) &
      + b*( q(i+2,j,4) + q(i-1,j,4) ) &
      + c*( q(i+3,j,4) + q(i-2,j,4) ) )

      
vf(i,j,1) = rh*uu
vf(i,j,2) = rh*uu*uu + pp
vf(i,j,3) = rh*uu*vv
vf(i,j,4) = (re+pp)*uu

end do
end do


!compute fluxes at cell interfaces:y

do j=0,ny
do i=0,nx

uu = g*(a*( u(i,j+1) + u(i,j) ) &
      + b*( u(i,j+2) + u(i,j-1) ) &
      + c*( u(i,j+3) + u(i,j-2) ) )
      
vv = g*(a*( v(i,j+1) + v(i,j) ) &
      + b*( v(i,j+2) + v(i,j-1) ) &
      + c*( v(i,j+3) + v(i,j-2) ) )
      
pp = g*(a*( p(i,j+1) + p(i,j) ) &
      + b*( p(i,j+2) + p(i,j-1) ) &
      + c*( p(i,j+3) + p(i,j-2) ) )

rh = g*(a*( q(i,j+1,1) + q(i,j,1) ) &
      + b*( q(i,j+2,1) + q(i,j-1,1) ) &
      + c*( q(i,j+3,1) + q(i,j-2,1) ) )

re = g*(a*( q(i,j+1,4) + q(i,j,4) ) &
      + b*( q(i,j+2,4) + q(i,j-1,4) ) &
      + c*( q(i,j+3,4) + q(i,j-2,4) ) )

      
vg(i,j,1) = rh*vv
vg(i,j,2) = rh*uu*vv 
vg(i,j,3) = rh*vv*vv + pp
vg(i,j,4) = (re+pp)*vv

end do
end do


! compute RHS contribution due to inviscid term (central difference)
do m=1,4
do j=1,ny
do i=1,nx

s(i,j,m) = - (vf(i,j,m)-vf(i-1,j,m))/dx &
           - (vg(i,j,m)-vg(i,j-1,m))/dy 

end do
end do
end do

deallocate(u,v,p,vf,vg)


!Add gravitational source term for RTI
if (iprob .eq. 3) then

	do j=1,ny
	do i=1,nx  
		s(i,j,3)=s(i,j,3) - q(i,j,1)  
		s(i,j,4)=s(i,j,4) - q(i,j,3)      
	end do
	end do

end if

return
end


!-----------------------------------------------------------------------------------!
!Computing Right Hand Side (RHS) for ILES (only inviscid term)
!-----------------------------------------------------------------------------------!
subroutine rhsILES(nx,ny,dx,dy,q,s)
implicit none
integer::nx,ny,i,j,m,iprob,ix
real*8 ::dx,dy
real*8 ::q(-2:nx+3,-2:ny+3,4),s(nx,ny,4)
real*8,allocatable:: u(:,:),uL(:,:),uR(:,:),hL(:,:),hR(:,:),rf(:,:)

common /problem/ iprob
common /fluxopt/ ix

!-----------------------------!
!Compute x-fluxes for all j
!-----------------------------!
allocate(u(-2:nx+3,4))
allocate(uL(0:nx,4),uR(0:nx,4),hL(0:nx,4),hR(0:nx,4),rf(0:nx,4))

do j=1,ny
	
	!assign q vector as u in x-direction
    do m=1,4
    do i=-2,nx+3
    u(i,m)=q(i,j,m)
    end do
    end do
    
	!-----------------------------!
	!Reconstruction scheme:
    !-----------------------------!
    !WENO5 construction
	call weno5(nx,u,uL,uR)
	
	!compute left and right fluxes
  	call xflux(nx,uL,hL)
	call xflux(nx,uR,hR)  
    
    !-----------------------------!
	!Riemann Solver: 
    !-----------------------------!
    if (ix.eq.1) then
    call rusanov_x(nx,u,uL,uR,hL,hR,rf)
    else if (ix.eq.2) then
    call roe_x(nx,uL,uR,hL,hR,rf)
    else if (ix.eq.3) then
    call hll_x(nx,uL,uR,hL,hR,rf)
    else if (ix.eq.4) then
    call ausm_x(nx,uL,uR,rf)  
    end if
	
    !-----------------------------!
	!Compute RHS contribution
    !-----------------------------!
	do m=1,4
	do i=1,nx 
	s(i,j,m)=-(rf(i,m)-rf(i-1,m))/dx     
	end do
    end do
    
end do

deallocate(u,uL,uR,hL,hR,rf)

!-----------------------------!
!Compute y-fluxes for all i
!-----------------------------!
allocate(u(-2:ny+3,4))
allocate(uL(0:ny,4),uR(0:ny,4),hL(0:ny,4),hR(0:ny,4),rf(0:ny,4))

do i=1,nx
	
	!assign q vector as u in y-direction
    do m=1,4
    do j=-2,ny+3
    u(j,m)=q(i,j,m)
    end do
    end do

    !-----------------------------!
	!Reconstruction scheme:
    !-----------------------------!
    !WENO5 construction
	call weno5(ny,u,uL,uR)
	
	!compute left and right fluxes
  	call yflux(ny,uL,hL)
	call yflux(ny,uR,hR)  


	!-----------------------------!
    !Riemann Solver: 
    !-----------------------------!   
    if (ix.eq.1) then
    call rusanov_y(ny,u,uL,uR,hL,hR,rf)
    else if (ix.eq.2) then
    call roe_y(ny,uL,uR,hL,hR,rf)
    else if (ix.eq.3) then
    call hll_y(ny,uL,uR,hL,hR,rf)
    else if (ix.eq.4) then
    call ausm_y(ny,uL,uR,rf)
    end if


    
	!-----------------------------!
	!Compute RHS contribution
    !-----------------------------!
	do m=1,4
	do j=1,ny 
	s(i,j,m)=s(i,j,m)-(rf(j,m)-rf(j-1,m))/dy     
	end do
    end do
       
end do
deallocate(u,uL,uR,hL,hR,rf)


!Add gravitational source term for RTI
if (iprob .eq. 3) then

	do j=1,ny
	do i=1,nx  
		s(i,j,3)=s(i,j,3) - q(i,j,1)  
		s(i,j,4)=s(i,j,4) - q(i,j,3)      
	end do
	end do

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
real*8 ::u(-2:n+3,4),uL(0:n,4),uR(0:n,4),hL(0:n,4),hR(0:n,4),rf(0:n,4)
real*8 ::l1,l2,l3,rad0,rad1,ps,p,a
real*8 ::gamma
    
common /fluids/ gamma

    
do i=0,n
  	
	!at point i
    p = (gamma-1.0d0)*(u(i,4)-0.5d0*(u(i,2)*u(i,2)/u(i,1)+u(i,3)*u(i,3)/u(i,1)))
	a = dsqrt(gamma*p/u(i,1)) 
    
	l1=dabs(u(i,2)/u(i,1))
	l2=dabs(u(i,2)/u(i,1) + a)
	l3=dabs(u(i,2)/u(i,1) - a)
	rad0 = max(l1,l2,l3)

    !at point i+1
    p = (gamma-1.0d0)*(u(i+1,4)-0.5d0*(u(i+1,2)*u(i+1,2)/u(i+1,1)+u(i+1,3)*u(i+1,3)/u(i+1,1)))
	a = dsqrt(gamma*p/u(i+1,1)) 
    
	l1=dabs(u(i+1,2)/u(i+1,1))
	l2=dabs(u(i+1,2)/u(i+1,1) + a)
	l3=dabs(u(i+1,2)/u(i+1,1) - a)
	rad1 = max(l1,l2,l3)
    
		!characteristic speed for Rusanov flux
		ps = max(rad0,rad1)

		!compute flux 
		do m=1,4  
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
real*8 ::u(-2:n+3,4),uL(0:n,4),uR(0:n,4),hL(0:n,4),hR(0:n,4),rf(0:n,4)
real*8 ::l1,l2,l3,rad0,rad1,ps,p,a
real*8 ::gamma
    
common /fluids/ gamma

    
do i=0,n
  	
	!at point i
    p = (gamma-1.0d0)*(u(i,4)-0.5d0*(u(i,2)*u(i,2)/u(i,1)+u(i,3)*u(i,3)/u(i,1)))
	a = dsqrt(gamma*p/u(i,1)) 
    
	l1=dabs(u(i,3)/u(i,1))
	l2=dabs(u(i,3)/u(i,1) + a)
	l3=dabs(u(i,3)/u(i,1) - a)
	rad0 = max(l1,l2,l3)

    !at point i+1
    p = (gamma-1.0d0)*(u(i+1,4)-0.5d0*(u(i+1,2)*u(i+1,2)/u(i+1,1)+u(i+1,3)*u(i+1,3)/u(i+1,1)))
	a = dsqrt(gamma*p/u(i+1,1)) 
    
	l1=dabs(u(i+1,3)/u(i+1,1))
	l2=dabs(u(i+1,3)/u(i+1,1) + a)
	l3=dabs(u(i+1,3)/u(i+1,1) - a)
	rad1 = max(l1,l2,l3)
    
		!characteristic speed for Rusanov flux
		ps = max(rad0,rad1)

		!compute flux 
		do m=1,4  
    	rf(i,m)=0.5d0*((hR(i,m)+hL(i,m)) - ps*(uR(i,m)-uL(i,m)))      
		end do
    
end do
    
return
end
!-----------------------------------------------------------------------------------!
!Riemann Solver: Roe flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine roe_x(n,uL,uR,hL,hR,rf)
implicit none
integer::n,m,i,ie
real*8 ::uL(0:n,4),uR(0:n,4),hL(0:n,4),hR(0:n,4),rf(0:n,4),cr(4)
real*8 ::gamma,delta,cfix
real*8 ::rhLL,uuLL,vvLL,eeLL,ppLL,hhLL,rhRR,uuRR,vvRR,eeRR,ppRR,hhRR
real*8 ::uu,vv,hh,aa,g1,g2,g3,g4,beta,teta,q1,q2,q3,q4
real*8 ::a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44
real*8 ::r11,r12,r13,r14,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44
real*8 ::l11,l12,l13,l14,l21,l22,l23,l24,l31,l32,l33,l34,l41,l42,l43,l44
    
common /fluids/ gamma
common /entropyfix/ cfix,ie

	!-----------------------------!
	!Roe flux in x-direction
    !-----------------------------!
do i=0,n

	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL	
	eeLL = uL(i,4)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL))
    hhLL = eeLL + ppLL/rhLL


	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	eeRR = uR(i,4)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR))
    hhRR = eeRR + ppRR/rhRR


	!Roe averages
	uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv)))

 
    !eigenvalues
    g1 = dabs(uu)
    g2 = dabs(uu)
    g3 = dabs(uu + aa)
    g4 = dabs(uu - aa)

	beta = 0.5d0/(aa*aa)
    teta = 0.5d0*(gamma-1.0d0)*(uu*uu+vv*vv)
     
	!entropy fix
    if (ie.eq.1) then
      
      	delta= 2.0d0*cfix*aa
      
        if (g1.lt.delta) g1 = (g1*g1 +delta*delta)/(2.0d0*delta)
    	if (g2.lt.delta) g2 = (g2*g2 +delta*delta)/(2.0d0*delta)
   		if (g3.lt.delta) g3 = (g3*g3 +delta*delta)/(2.0d0*delta)
    	if (g4.lt.delta) g4 = (g4*g4 +delta*delta)/(2.0d0*delta)
	
    end if
    
    !right eigenvectors
    r11 = 1.0d0
    r12 = 0.0d0
    r13 = beta
    r14 = beta

    r21 = uu
    r22 = 0.0d0
    r23 = beta*(uu+aa)
    r24 = beta*(uu-aa)  
    
	r31 = vv
    r32 =-1.0d0
    r33 = beta*vv
    r34 = beta*vv

    r41 = teta/(gamma-1.0d0)  
    r42 =-vv
    r43 = beta*(hh+uu*aa)
    r44 = beta*(hh-uu*aa)    

	!left eigenvectors
    l11 = 1.0d0-teta/(aa*aa)
    l12 = (gamma-1.0d0)*uu/(aa*aa)
    l13 = (gamma-1.0d0)*vv/(aa*aa)
	l14 =-(gamma-1.0d0)/(aa*aa)

    l21 = vv
    l22 = 0.0d0
    l23 =-1.0d0
    l24 = 0.0d0

    l31 = teta - uu*aa
    l32 = aa - (gamma-1.0d0)*uu
    l33 =-(gamma-1.0d0)*vv
    l34 = (gamma-1.0d0)    
      
    l41 = teta + uu*aa
    l42 =-aa - (gamma-1.0d0)*uu
    l43 =-(gamma-1.0d0)*vv
    l44 = (gamma-1.0d0) 

    
    !matrix multiplication
    a11 = g1*l11*r11 + g2*l21*r12 + g3*l31*r13 + g4*l41*r14
    a12 = g1*l12*r11 + g2*l22*r12 + g3*l32*r13 + g4*l42*r14
    a13 = g1*l13*r11 + g2*l23*r12 + g3*l33*r13 + g4*l43*r14
    a14 = g1*l14*r11 + g2*l24*r12 + g3*l34*r13 + g4*l44*r14

    a21 = g1*l11*r21 + g2*l21*r22 + g3*l31*r23 + g4*l41*r24
    a22 = g1*l12*r21 + g2*l22*r22 + g3*l32*r23 + g4*l42*r24
    a23 = g1*l13*r21 + g2*l23*r22 + g3*l33*r23 + g4*l43*r24
    a24 = g1*l14*r21 + g2*l24*r22 + g3*l34*r23 + g4*l44*r24

	a31 = g1*l11*r31 + g2*l21*r32 + g3*l31*r33 + g4*l41*r34
    a32 = g1*l12*r31 + g2*l22*r32 + g3*l32*r33 + g4*l42*r34
    a33 = g1*l13*r31 + g2*l23*r32 + g3*l33*r33 + g4*l43*r34
    a34 = g1*l14*r31 + g2*l24*r32 + g3*l34*r33 + g4*l44*r34

    a41 = g1*l11*r41 + g2*l21*r42 + g3*l31*r43 + g4*l41*r44
    a42 = g1*l12*r41 + g2*l22*r42 + g3*l32*r43 + g4*l42*r44
    a43 = g1*l13*r41 + g2*l23*r42 + g3*l33*r43 + g4*l43*r44
    a44 = g1*l14*r41 + g2*l24*r42 + g3*l34*r43 + g4*l44*r44
    
	q1 = uR(i,1)-uL(i,1)
    q2 = uR(i,2)-uL(i,2)
    q3 = uR(i,3)-uL(i,3)
    q4 = uR(i,4)-uL(i,4)
    
    cr(1) = q1*a11 + q2*a12 + q3*a13 + q4*a14
    cr(2) = q1*a21 + q2*a22 + q3*a23 + q4*a24
    cr(3) = q1*a31 + q2*a32 + q3*a33 + q4*a34
    cr(4) = q1*a41 + q2*a42 + q3*a43 + q4*a44
		
		!compute flux in x-direction
    	do m=1,4	
		rf(i,m)=0.5d0*(hR(i,m)+hL(i,m)) - 0.5d0*cr(m)     
    	end do
        
end do

return
end

!-----------------------------------------------------------------------------------!
!Riemann Solver: Roe flux
!uL: weno reconstracted values at left state (positive)
!uR: weno reconstracted values at rigth state (negative)
!hL: flux at left state
!hR: flux at right state
!rf: flux
!-----------------------------------------------------------------------------------!
subroutine roe_y(n,uL,uR,hL,hR,rf)
implicit none

integer::n,m,j,ie
real*8 ::uL(0:n,4),uR(0:n,4),hL(0:n,4),hR(0:n,4),rf(0:n,4),cr(4)
real*8 ::gamma,delta,cfix
real*8 ::rhLL,uuLL,vvLL,eeLL,ppLL,hhLL,rhRR,uuRR,vvRR,eeRR,ppRR,hhRR
real*8 ::uu,vv,hh,aa,g1,g2,g3,g4,beta,teta
real*8 ::q1,q2,q3,q4
real*8 ::a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44
real*8 ::r11,r12,r13,r14,r21,r22,r23,r24,r31,r32,r33,r34,r41,r42,r43,r44
real*8 ::l11,l12,l13,l14,l21,l22,l23,l24,l31,l32,l33,l34,l41,l42,l43,l44
    
common /fluids/ gamma
!common /entropyfix/ cfix,ie

	!-----------------------------!
	!Roe flux in y-direction
    !-----------------------------!

do j=0,n

	!Left and right states:
	rhLL = uL(j,1)
	uuLL = uL(j,2)/rhLL	
	vvLL = uL(j,3)/rhLL	
	eeLL = uL(j,4)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL))
    hhLL = eeLL + ppLL/rhLL

	rhRR = uR(j,1)
	uuRR = uR(j,2)/rhRR
	vvRR = uR(j,3)/rhRR
	eeRR = uR(j,4)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR))
    hhRR = eeRR + ppRR/rhRR


	!Roe averages
	uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
	aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv)))
  
    !eigenvalues
    g1 = dabs(vv)
    g2 = dabs(vv)
    g3 = dabs(vv + aa)
    g4 = dabs(vv - aa)

	beta = 0.5d0/(aa*aa)
    teta = 0.5d0*(gamma-1.0d0)*(uu*uu+vv*vv)
    
	!entropy fix
    if (ie.eq.1) then
      
      	delta= 2.0d0*cfix*aa
      
        if (g1.lt.delta) g1 = (g1*g1 +delta*delta)/(2.0d0*delta)
    	if (g2.lt.delta) g2 = (g2*g2 +delta*delta)/(2.0d0*delta)
    	if (g3.lt.delta) g3 = (g3*g3 +delta*delta)/(2.0d0*delta)
    	if (g4.lt.delta) g4 = (g4*g4 +delta*delta)/(2.0d0*delta)
	         
    end if

    !right eigenvectors
    r11 = 1.0d0
    r12 = 0.0d0
    r13 = beta
    r14 = beta

	r21 = uu
    r22 = 1.0d0
    r23 = beta*uu
    r24 = beta*uu
    
    r31 = vv
    r32 = 0.0d0
    r33 = beta*(vv+aa)
    r34 = beta*(vv-aa)  
    
    r41 = teta/(gamma-1.0d0)  
    r42 = uu
    r43 = beta*(hh+vv*aa)
    r44 = beta*(hh-vv*aa)    

	!left eigenvectors
    l11 = 1.0d0-teta/(aa*aa)
    l12 = (gamma-1.0d0)*uu/(aa*aa)
    l13 = (gamma-1.0d0)*vv/(aa*aa)
	l14 =-(gamma-1.0d0)/(aa*aa)

    l21 =-uu
    l22 = 1.0d0
    l23 = 0.0d0
    l24 = 0.0d0

    l31 = teta - vv*aa
    l32 =-(gamma-1.0d0)*uu
    l33 = aa - (gamma-1.0d0)*vv
    l34 = (gamma-1.0d0)    
      
    l41 = teta + vv*aa
    l42 =-(gamma-1.0d0)*uu 
    l43 =-aa - (gamma-1.0d0)*vv
    l44 = (gamma-1.0d0) 

    
    !matrix multiplication
    a11 = g1*l11*r11 + g2*l21*r12 + g3*l31*r13 + g4*l41*r14
    a12 = g1*l12*r11 + g2*l22*r12 + g3*l32*r13 + g4*l42*r14
    a13 = g1*l13*r11 + g2*l23*r12 + g3*l33*r13 + g4*l43*r14
    a14 = g1*l14*r11 + g2*l24*r12 + g3*l34*r13 + g4*l44*r14

    a21 = g1*l11*r21 + g2*l21*r22 + g3*l31*r23 + g4*l41*r24
    a22 = g1*l12*r21 + g2*l22*r22 + g3*l32*r23 + g4*l42*r24
    a23 = g1*l13*r21 + g2*l23*r22 + g3*l33*r23 + g4*l43*r24
    a24 = g1*l14*r21 + g2*l24*r22 + g3*l34*r23 + g4*l44*r24

	a31 = g1*l11*r31 + g2*l21*r32 + g3*l31*r33 + g4*l41*r34
    a32 = g1*l12*r31 + g2*l22*r32 + g3*l32*r33 + g4*l42*r34
    a33 = g1*l13*r31 + g2*l23*r32 + g3*l33*r33 + g4*l43*r34
    a34 = g1*l14*r31 + g2*l24*r32 + g3*l34*r33 + g4*l44*r34

    a41 = g1*l11*r41 + g2*l21*r42 + g3*l31*r43 + g4*l41*r44
    a42 = g1*l12*r41 + g2*l22*r42 + g3*l32*r43 + g4*l42*r44
    a43 = g1*l13*r41 + g2*l23*r42 + g3*l33*r43 + g4*l43*r44
    a44 = g1*l14*r41 + g2*l24*r42 + g3*l34*r43 + g4*l44*r44
    
	q1 = uR(j,1)-uL(j,1)
    q2 = uR(j,2)-uL(j,2)
    q3 = uR(j,3)-uL(j,3)
    q4 = uR(j,4)-uL(j,4)
    
    cr(1) = q1*a11 + q2*a12 + q3*a13 + q4*a14
    cr(2) = q1*a21 + q2*a22 + q3*a23 + q4*a24
    cr(3) = q1*a31 + q2*a32 + q3*a33 + q4*a34
    cr(4) = q1*a41 + q2*a42 + q3*a43 + q4*a44
		
		!compute flux in y-direction
    	do m=1,4	
		rf(j,m)=0.5d0*(hR(j,m)+hL(j,m)) - 0.5d0*cr(m)     
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
integer::n,m,i!,iw
real*8 ::uL(0:n,4),uR(0:n,4),hL(0:n,4),hR(0:n,4),rf(0:n,4)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,eeLL,ppLL,aaLL,rhRR,uuRR,vvRR,eeRR,ppRR,aaRR
real*8 ::SL,SR
!real*8 ::uu,vv,hh,aa,eta,dd,rh,px,qLL,qRR,hhLL,hhRR

common /fluids/ gamma
!common /waveopt/ iw

do i=0,n
  	
	!Left and right states:
	rhLL = uL(i,1)
	uuLL = uL(i,2)/rhLL	
	vvLL = uL(i,3)/rhLL	
	eeLL = uL(i,4)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL))
!    hhLL = eeLL + ppLL/rhLL
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	eeRR = uR(i,4)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR))
!    hhRR = eeRR + ppRR/rhRR
    aaRR = dsqrt(gamma*ppRR/rhRR)

	!Compute SL and SR
!	if(iw.eq.1) then

		SL = min(uuLL,uuRR) - max(aaLL,aaRR)
		SR = max(uuLL,uuRR) + max(aaLL,aaRR)

!	else if(iw.eq.2) then

!		uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv)))		
		
!		SL = uu - aa
!		SR = uu + aa

!	else if(iw.eq.3) then

!		eta = 0.5d0*dsqrt(rhLL)*dsqrt(rhRR)/(dsqrt(rhLL) + dsqrt(rhRR))**2
	
!		dd = dsqrt( (dsqrt(rhLL)*(aaLL**2) + dsqrt(rhRR)*(aaRR**2))/(dsqrt(rhLL) + dsqrt(rhRR)) &
!		           + eta* (uuRR-uuLL)**2 )

!		SL = uu - dd
!		SR = uu + dd

!	else if(iw.eq.4) then

!		rh = 0.5d0*(rhLL+rhRR)
!		aa = 0.5d0*(aaLL+aaRR)
		
!		px = 0.5d0*((ppLL+ppRR)-(uuRR-uuLL)*rh*aa)
		
!		if(px.le.ppLL) then
!		qLL = 1.0d0
!		else
!		qLL = dsqrt(1.0d0+(gamma+1.0d0)/(2.0d0*gamma)*(px/ppLL - 1.0d0))
!		end if

!		if(px.le.ppRR) then
!		qRR = 1.0d0
!		else
!		qRR = dsqrt(1.0d0+(gamma+1.0d0)/(2.0d0*gamma)*(px/ppRR - 1.0d0))
!		end if
		
!		SL = uuLL - aaLL*qLL
!		SR = uuRR + aaRR*qRR

!	else

!		uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv)))

!		SL = min(uuLL-aaLL, uu-aa)
!		SR = max(uuRR+aaRR, uu+aa)
		 
!	end if

	!compute HLL flux in x-direction
	if(SL.ge.0.0d0) then
		do m=1,4  
    	rf(i,m)=hL(i,m)     
		end do	
	else if (SR.le.0.0d0) then
		do m=1,4  
    	rf(i,m)=hR(i,m)     
		end do	
	else 
		do m=1,4  
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
integer::n,m,j!,iw
real*8 ::uL(0:n,4),uR(0:n,4),hL(0:n,4),hR(0:n,4),rf(0:n,4)
real*8 ::gamma
real*8 ::rhLL,uuLL,vvLL,eeLL,ppLL,aaLL,rhRR,uuRR,vvRR,eeRR,ppRR,aaRR
real*8 ::SL,SR
!real*8 ::uu,vv,hh,aa,eta,dd,rh,px,qLL,qRR,hhLL,hhRR

common /fluids/ gamma
!common /waveopt/ iw

do j=0,n
  	
	!Left and right states:
	rhLL = uL(j,1)
	uuLL = uL(j,2)/rhLL	
	vvLL = uL(j,3)/rhLL	
	eeLL = uL(j,4)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL))
!    hhLL = eeLL + ppLL/rhLL
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(j,1)
	uuRR = uR(j,2)/rhRR
	vvRR = uR(j,3)/rhRR
	eeRR = uR(j,4)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR))
!   hhRR = eeRR + ppRR/rhRR
    aaRR = dsqrt(gamma*ppRR/rhRR)

	!Compute SL and SR
!	if(iw.eq.1) then

		SL = min(vvLL,vvRR) - max(aaLL,aaRR)
		SR = max(vvLL,vvRR) + max(aaLL,aaRR)

!	else if(iw.eq.2) then

!		uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv)))		
		
!		SL = vv - aa
!		SR = vv + aa

!	else if(iw.eq.3) then

!		eta = 0.5d0*dsqrt(rhLL)*dsqrt(rhRR)/(dsqrt(rhLL) + dsqrt(rhRR))**2
	
!		dd = dsqrt( (dsqrt(rhLL)*(aaLL**2) + dsqrt(rhRR)*(aaRR**2))/(dsqrt(rhLL) + dsqrt(rhRR)) &
!		           + eta* (uuRR-uuLL)**2 )

!		SL = vv - dd
!		SR = vv + dd

!	else if(iw.eq.4) then

!		rh = 0.5d0*(rhLL+rhRR)
!		aa = 0.5d0*(aaLL+aaRR)
		
!		px = 0.5d0*((ppLL+ppRR)-(vvRR-vvLL)*rh*aa)
		
!		if(px.le.ppLL) then
!		qLL = 1.0d0
!		else
!		qLL = dsqrt(1.0d0+(gamma+1.0d0)/(2.0d0*gamma)*(px/ppLL - 1.0d0))
!		end if

!		if(px.le.ppRR) then
!		qRR = 1.0d0
!		else
!		qRR = dsqrt(1.0d0+(gamma+1.0d0)/(2.0d0*gamma)*(px/ppRR - 1.0d0))
!		end if
		
!		SL = vvLL - aaLL*qLL
!		SR = vvRR + aaRR*qRR


!	else

!		uu = (dsqrt(rhLL)*uuLL + dsqrt(rhRR)*uuRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		vv = (dsqrt(rhLL)*vvLL + dsqrt(rhRR)*vvRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		hh = (dsqrt(rhLL)*hhLL + dsqrt(rhRR)*hhRR)/(dsqrt(rhLL) + dsqrt(rhRR))
!		aa = dsqrt((gamma-1.0d0)*(hh - 0.5d0*(uu*uu+vv*vv)))

!		SL = min(vvLL-aaLL, vv-aa)
!		SR = max(vvRR+aaRR, vv+aa)

!	end if



	!compute HLL flux in y-direction
	if(SL.ge.0.0d0) then
		do m=1,4  
    	rf(j,m)=hL(j,m)     
		end do	
	else if (SR.le.0.0d0) then
		do m=1,4  
    	rf(j,m)=hR(j,m)     
		end do	
	else 
		do m=1,4  
    	rf(j,m)=(SR*hL(j,m)-SL*hR(j,m)+SL*SR*(uR(j,m)-uL(j,m)))/(SR-SL)     
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
real*8 ::uL(0:n,4),uR(0:n,4),rf(0:n,4)
real*8 ::rhLL,uuLL,vvLL,eeLL,ppLL,hhLL,aaLL,rhRR,uuRR,vvRR,eeRR,ppRR,hhRR,aaRR
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
	eeLL = uL(i,4)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL))
    hhLL = eeLL + ppLL/rhLL
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(i,1)
	uuRR = uR(i,2)/rhRR
	vvRR = uR(i,3)/rhRR
	eeRR = uR(i,4)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR))
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
	rf(i,4)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*hhRR) + (M12+dabs(M12))*(rhLL*aaLL*hhLL))   
    
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
integer::n,j
real*8 ::uL(0:n,4),uR(0:n,4),rf(0:n,4)
real*8 ::rhLL,uuLL,vvLL,eeLL,ppLL,hhLL,aaLL,rhRR,uuRR,vvRR,eeRR,ppRR,hhRR,aaRR
real*8 ::MLL,pLL,MRR,pRR,M12,p12
real*8 ::gamma
    
common /fluids/ gamma

    !-----------------------------!
	!AUSM flux in y-direction
    !-----------------------------!

do j=0,n
  	
	!Left and right states:
	rhLL = uL(j,1)
	uuLL = uL(j,2)/rhLL	
	vvLL = uL(j,3)/rhLL	
	eeLL = uL(j,4)/rhLL
    ppLL = (gamma-1.0d0)*(eeLL*rhLL - 0.5d0*rhLL*(uuLL*uuLL+vvLL*vvLL))
    hhLL = eeLL + ppLL/rhLL
    aaLL = dsqrt(gamma*ppLL/rhLL)

	rhRR = uR(j,1)
	uuRR = uR(j,2)/rhRR
	vvRR = uR(j,3)/rhRR
	eeRR = uR(j,4)/rhRR
    ppRR = (gamma-1.0d0)*(eeRR*rhRR - 0.5d0*rhRR*(uuRR*uuRR+vvRR*vvRR))
    hhRR = eeRR + ppRR/rhRR
    aaRR = dsqrt(gamma*ppRR/rhRR)
    
	if (dabs(vvLL/aaLL).le.1.0d0) then
	MLL = 0.25d0*(vvLL/aaLL + 1.0d0)**2
	pLL = 0.25d0*ppLL*((vvLL/aaLL + 1.0d0)**2)*(2.0d0-vvLL/aaLL)
	!pLL = 0.5d0*ppLL*(1.0d0+vvLL/aaLL)
	else
	MLL = 0.5d0*(vvLL/aaLL + dabs(vvLL/aaLL))
	pLL = 0.5d0*ppLL*(vvLL/aaLL + dabs(vvLL/aaLL))/(vvLL/aaLL)
	end if

	if (dabs(vvRR/aaRR).le.1.0d0) then
	MRR =-0.25d0*(vvRR/aaRR - 1.0d0)**2
	pRR = 0.25d0*ppRR*((vvRR/aaRR - 1.0d0)**2)*(2.0d0+vvRR/aaRR)
	!pRR = 0.5d0*ppRR*(1.0d0-vvRR/aaRR)
	else
	MRR = 0.5d0*(vvRR/aaRR - dabs(vvRR/aaRR))
	pRR = 0.5d0*ppRR*(vvRR/aaRR - dabs(vvRR/aaRR))/(vvRR/aaRR)
	end if

	M12 = MLL + MRR
	p12 = pLL + pRR

	!compute flux in y-direction		
    rf(j,1)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR) + (M12+dabs(M12))*(rhLL*aaLL)) 
	rf(j,2)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*uuRR) + (M12+dabs(M12))*(rhLL*aaLL*uuLL)) 
	rf(j,3)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*vvRR) + (M12+dabs(M12))*(rhLL*aaLL*vvLL)) + p12
	rf(j,4)=0.5d0*((M12-dabs(M12))*(rhRR*aaRR*hhRR) + (M12+dabs(M12))*(rhLL*aaLL*hhLL)) 
		
end do	

return
end    



!-----------------------------------------------------------------------------------!
!5th order WENO reconstruction
!-----------------------------------------------------------------------------------!
subroutine weno5(n,q,qL,qR)
implicit none
integer::n,m,i
real*8 ::q(-2:n+3,4),qL(0:n,4),qR(0:n,4)
real*8 ::eps,a,b,c,h,g,a1,a2,a3,w1,w2,w3,q1,q2,q3
real*8,allocatable ::b1(:),b2(:),b3(:)


common /weno_constant/ eps

    
    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0
    
    a = 1.0d0/10.0d0
    b = 3.0d0/5.0d0
    c = 3.0d0/10.0d0

!compute beta
allocate(b1(0:n+1))
allocate(b2(0:n+1))
allocate(b3(0:n+1))

do m=1,4
  
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

    a1 = a/(eps+b1(i))**2
    a2 = b/(eps+b2(i))**2
    a3 = c/(eps+b3(i))**2

    w1 = a1/(a1+a2+a3)
    w2 = a2/(a1+a2+a3)
    w3 = a3/(a1+a2+a3)
    
	q1 = g*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))
	q2 = g*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))
	q3 = g*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m))

		
	qL(i,m) = w1*q1 + w2*q2 + w3*q3


    !negative reconstruction at i+1/2
  
    a1 = c/(eps+b1(i+1))**2
    a2 = b/(eps+b2(i+1))**2
    a3 = a/(eps+b3(i+1))**2

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
!Time Step
!-----------------------------------------------------------------------------------!
subroutine timestep(nx,ny,dx,dy,cfl,dt,q)
implicit none
integer::nx,ny,i,j
real*8 ::dt,cfl,gamma,dx,dy,smx,smy,radx,rady,p,a,l1,l2,l3
real*8 ::q(-2:nx+3,-2:ny+3,4)

common /fluids/ gamma

!Spectral radius of Jacobian
smx = 0.0d0
smy = 0.0d0

do j=1,ny
do i=1,nx

p = (gamma-1.0d0)*(q(i,j,4)-0.5d0*(q(i,j,2)*q(i,j,2)/q(i,j,1)+q(i,j,3)*q(i,j,3)/q(i,j,1)))
a = dsqrt(gamma*p/q(i,j,1)) 
 
!in-x direction  
l1=dabs(q(i,j,2)/q(i,j,1))
l2=dabs(q(i,j,2)/q(i,j,1) + a)
l3=dabs(q(i,j,2)/q(i,j,1) - a)
radx = max(l1,l2,l3)

!in-y direction  
l1=dabs(q(i,j,3)/q(i,j,1))
l2=dabs(q(i,j,3)/q(i,j,1) + a)
l3=dabs(q(i,j,3)/q(i,j,1) - a)
rady = max(l1,l2,l3)

if (radx.gt.smx) smx = radx
if (rady.gt.smy) smy = rady
    
end do
end do

dt = min(cfl*dx/smx,cfl*dy/smy)

return 
end

!-----------------------------------------------------------------------------------!
!History: compute domain integrated total energy
!-----------------------------------------------------------------------------------!
subroutine history(nx,ny,q,np,te)
implicit none
integer::nx,ny,i,j,np
real*8 ::u,v,te
real*8 ::q(-2:nx+3,-2:ny+3,4)

te = 0.0d0
do j=1,ny
do i=1,nx
u=q(i,j,2)/q(i,j,1)
v=q(i,j,3)/q(i,j,1)
te = te + 0.5d0*(u*u + v*v)
end do
end do
te = te/dfloat(nx*ny*np)

return 
end



!-----------------------------------------------------------------------------------!
!Computing x-fluxes from conserved quantities 
!-----------------------------------------------------------------------------------!
subroutine xflux(nx,u,f)
implicit none
integer::nx,i
real*8::gamma,p
real*8::u(0:nx,4),f(0:nx,4)

common /fluids/ gamma

do i=0,nx
p = (gamma-1.0d0)*(u(i,4)-0.5d0*(u(i,2)*u(i,2)/u(i,1)+u(i,3)*u(i,3)/u(i,1)))
f(i,1) = u(i,2)
f(i,2) = u(i,2)*u(i,2)/u(i,1) + p
f(i,3) = u(i,2)*u(i,3)/u(i,1)
f(i,4) = (u(i,4)+ p)*u(i,2)/u(i,1)

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
real*8::u(0:ny,4),g(0:ny,4)

common /fluids/ gamma

do j=0,ny
p = (gamma-1.0d0)*(u(j,4)-0.5d0*(u(j,2)*u(j,2)/u(j,1)+u(j,3)*u(j,3)/u(j,1)))
g(j,1) = u(j,3)
g(j,2) = u(j,3)*u(j,2)/u(j,1) 
g(j,3) = u(j,3)*u(j,3)/u(j,1) + p
g(j,4) = (u(j,4)+ p)*u(j,3)/u(j,1)
end do

return
end

!-----------------------------------------------------------------------------------!
!Perona Malik Anisotropic Diffusion
!-----------------------------------------------------------------------------------!
subroutine PMAD(nx,ny,q,dt,dx)
implicit none
integer::nx,ny,i,j,m,npst,t
real*8 ::dt,kappa,dx,capK
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8,allocatable:: u(:,:,:),v(:,:,:),c(:,:,:)

common /PMconstant/ kappa,capK,npst

allocate(u(-2:nx+3,-2:ny+3,4))
allocate(v(-2:nx+3,-2:ny+3,4))
allocate(c(-2:nx+3,-2:ny+3,4))

do t = 1,npst

!Update BC
call bcs(nx,ny,q)


do m=1,4
do j=-2,ny+3
do i=-2,nx+3
u(i,j,m) = q(i,j,m)
end do
end do
end do

!Need to find value of c in y direction
!using original kernel
do m=1,4
do j=1,ny
do i=1,nx
c(i,j,m) = 1.0d0/(1+dabs((q(i,j+1,m)-q(i,j-1,m))/(2.0*dx*capK))**2)
end do
end do
end do

call bcs(nx,ny,c)

!filter in y
do m=1,4
do j=1,ny
do i=1,nx
v(i,j,m) = u(i,j,m) + kappa*dt/(2.0d0*dx**2)*((c(i,j,m)+c(i,j+1,m))*&
			&(u(i,j+1,m)-u(i,j,m))-(c(i,j-1,m)+c(i,j,m))*(u(i,j,m)-u(i,j-1,m)))
end do
end do
end do

call bcs(nx,ny,v)

!Need to find value of c in x direction
!using original kernel
do m=1,4
do j=1,ny
do i=1,nx
c(i,j,m) = 1.0d0/(1+dabs((v(i+1,j,m)-v(i-1,j,m))/(2.0*dx*capK))**2)
end do
end do
end do

call bcs(nx,ny,c)

!filter in x
do m=1,4
do j=1,ny
do i=1,nx
q(i,j,m) = v(i,j,m) + kappa*dt/(2.0d0*dx**2)*((c(i,j,m)+c(i+1,j,m))*&
			&(v(i+1,j,m)-v(i,j,m))-(c(i-1,j,m)+c(i,j,m))*(v(i,j,m)-v(i-1,j,m)))
end do
end do
end do

call bcs(nx,ny,q)

end do

deallocate(u,v,c)


return
end


!-----------------------------------------------------------------------------------!
!Filtering Bogeys Shock Filter
!-----------------------------------------------------------------------------------!
subroutine shockfilterSF7(nx,ny,q)
implicit none
integer::nx,ny,i,j,m
real*8 ::gamma,c1,c0,rth,eps,dpmag,rr
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8,allocatable:: p(:,:),dp(:,:),dscx(:,:,:),dscy(:,:,:)
real*8,allocatable:: sigx(:,:),sigy(:,:),qtemp(:,:,:),sigxf(:,:),sigyf(:,:)

common /fluids/ gamma
common /shock_trashold/ rth

c1 = -0.25d0
c0 = -c1
eps = 1.0d-16

!Update BC
call bcs(nx,ny,q)

allocate(qtemp(-2:nx+3,-2:ny+3,4))
allocate(p(-2:nx+3,-2:ny+3))
allocate(dp(-2:nx+3,-2:ny+3))
allocate(sigx(-2:nx+3,-2:ny+3)) 
allocate(sigy(-2:nx+3,-2:ny+3))	

allocate(sigxf(-2:nx+3,-2:ny+3))	!Faces 
allocate(sigyf(-2:nx+3,-2:ny+3))	!Faces	
allocate(dscx(-2:nx+3,-2:ny+3,4))	!Faces
allocate(dscy(-2:nx+3,-2:ny+3,4))	!Faces

!--------------------------------------------------------------------------------------
!Filtering in y direction
!--------------------------------------------------------------------------------------

do j=-2,ny+3
do i=1,nx
  p(i,j) = (gamma-1.0d0)*(q(i,j,4)-0.5d0*(q(i,j,2)*q(i,j,2)/q(i,j,1)+q(i,j,3)*q(i,j,3)/q(i,j,1)))
end do
end do

do j=-1,ny+2
do i=1,nx
  dp(i,j) = (-p(i,j+1)+2.0d0*p(i,j)-p(i,j-1))/4.0d0
end do
end do

do j=0,ny+1
do i=1,nx
  dpmag = 0.5d0*((dp(i,j)-dp(i,j+1))**2+(dp(i,j)-dp(i,j-1))**2)
  rr = dpmag/(p(i,j)*p(i,j)) + eps
  sigy(i,j) = 0.50d0*(1.0d0-rth/rr+dabs(1.0d0-rth/rr))
end do
end do

do j=0,ny
do i=1,nx
  sigyf(i,j) = 0.5d0*(sigy(i,j)+sigy(i,j+1))
end do
end do

do m =1,4
do j = 0,ny
do i = 1,nx
	dscy(i,j,m) = c0*q(i,j,m)+c1*q(i,j+1,m)    
end do
end do
end do

do m =1,4
do j = 1,ny
do i = 1,nx
	qtemp(i,j,m) = q(i,j,m)-(sigyf(i,j)*dscy(i,j,m)-sigyf(i,j-1)*dscy(i,j-1,m))
end do
end do
end do

call bcs(nx,ny,qtemp)

!--------------------------------------------------------------------------------------
!Filtering in x direction
!--------------------------------------------------------------------------------------

do j=1,ny
do i=-2,nx+3
p(i,j) = (gamma-1.0d0)*(qtemp(i,j,4)-0.5d0*(qtemp(i,j,2)*qtemp(i,j,2)/qtemp(i,j,1) &
        + qtemp(i,j,3)*qtemp(i,j,3)/qtemp(i,j,1)))
end do
end do


do j=1,ny
do i=-1,nx+2
  dp(i,j) = (-p(i+1,j)+2.0d0*p(i,j)-p(i-1,j))/4.0d0
end do
end do

do j=1,ny
do i=0,nx+1
  dpmag = 0.5d0*((dp(i,j)-dp(i+1,j))**2 + (dp(i,j)-dp(i-1,j))**2)
  rr = dpmag/(p(i,j)*p(i,j)) + eps
  sigx(i,j) = 0.50d0*(1.0d0-rth/rr+dabs(1.0d0-rth/rr))
end do
end do

do j=1,ny
do i=0,nx
  sigxf(i,j) = 0.5d0*(sigx(i,j)+sigx(i+1,j))
end do
end do

do m =1,4
do j = 1,ny
do i = 0,nx
	dscx(i,j,m) = c0*qtemp(i,j,m)+c1*qtemp(i+1,j,m)       
end do
end do
end do


do m =1,4
do j = 1,ny
do i = 1,nx
	q(i,j,m) = qtemp(i,j,m)-(sigxf(i,j)*dscx(i,j,m)-sigxf(i-1,j)*dscx(i-1,j,m))
end do
end do
end do


call bcs(nx,ny,q)


deallocate(p,dp,sigx,sigy,sigxf,sigyf,dscx,dscy,qtemp)

return
end        

