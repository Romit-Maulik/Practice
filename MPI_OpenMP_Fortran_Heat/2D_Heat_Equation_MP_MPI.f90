!------------------------------------------------------------------------------!
!		>>> 2D Heat Equation tutorial - Hybrid OpenMP/MPI >>> 
!------------------------------------------------------------------------------!                                
!------------------------------------------------------------------------------!
!Parallel Code Implementation with MPI
!To run in Cowboy Cluster (OSU): 
!          	module load openmpi-1.4/intel
!          	mpif90 -openmp 2D_Heat_Equation_MP_MPI.f90
!		   	qsub mexp.job (mbatch.job or mlong.job)
!------------------------------------------------------------------------------!
!Schemes: 	2th-order central scheme - space
!         	3rd order TVD-RK for time integration
!------------------------------------------------------------------------------!
!BCs: 		Periodic boundary conditions for top and bottom walls (y direction)
!BCs: 		No-slip (i.e. u=0) for left and right walls (in x direction)
!------------------------------------------------------------------------------!
!Domain: 	Square [x:0,2pi]*[y:0,2pi]
!------------------------------------------------------------------------------!
!Authors:	CFDLab.org, cfdlab.osu@gmail.com, romit.maulik@okstate.edu
!			Oklahoma State University, Stillwater			
!------------------------------------------------------------------------------!
!Updated: 	Dec 2, 2017 
!------------------------------------------------------------------------------!
program heatmpmpi
implicit none

include 'mpif.h'	!MPI header file

!Physics related variables
integer:: nx,ny,nt,nsnap,snapshot
integer :: i,j,k
real*8 :: dx,dy,dt,lx,ly,cfl,tmax
real*8 :: pi,x0,y0,alpha,t,snap_time,a,b
real*8,allocatable :: u(:,:),x(:),y(:)
real*8,allocatable :: u1(:,:),s(:,:)

!MPI related variables
integer               :: myid
integer               :: np
integer               :: ierr
integer 			  :: jip,nsend,ny1,ny_global,ic
real*8 				  :: y_min

common /heatcoeff/ alpha
common /lengths /lx,ly
common /mpivars/myid,nsend,np

!Reading input file
open(10,file='input.txt')
read(10,*)nx
read(10,*)ny_global
read(10,*)cfl
read(10,*)tmax
read(10,*)alpha
read(10,*)nsnap
close(10)

!----------------------------------------------------------------------!
! MPI initialize
!----------------------------------------------------------------------!
      call MPI_INIT(ierr) 

      call MPI_Comm_size(MPI_COMM_WORLD, np, ierr) 

      call MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)

!----------------------------------------------------------------------!
! Domain decomposition
! We only consider to decompose in z-direction (simplify)
!----------------------------------------------------------------------!


! Local array dimensions
  ny1 = int(ny_global/np)
  jip = ny_global - np*ny1 - 1
  
    !load balancing
	if(myid.le.jip) then
    ny=ny1+1
    else
    ny=ny1
    end if 

!----------------------------------------------------------------------!
!Global domain:
!y is the streching domain for channel
!----------------------------------------------------------------------!
	pi = 4.0d0*datan(1.0d0)

	!equidistant grid in x
	lx = 2.0d0*pi
	dx = lx/dfloat(nx)

	!equidistant grid in y
	ly = 2.0d0*pi
	dy = ly/dfloat(ny_global)


x0 = 0.0d0
y0 = 0.0d0

if(myid.le.jip) then
y_min= - 0.5d0*dy  + dfloat(myid)*dy*dfloat(ny) 
else
y_min= - 0.5d0*dy  + dfloat(myid)*dy*dfloat(ny) + dfloat(jip+1)*dy  
end if


!total send and receive points
nsend = (nx+2)!+- 1 point stencil for simple FTCS

! Local grid  
! cell-centered grid points:
allocate(x(0:nx+1))
allocate(y(0:ny+1))

!x coordinate
	do i =0, nx+1
    x(i) = x0 - 0.5d0*dx + dfloat(i)*dx   
	end do

!y coordinate
	do j =0, ny+1 
    y(j) = y0 + y_min + dfloat(j)*dy
	end do

!----------------------------------------------------------------------!
!Problem definition and initialization
!----------------------------------------------------------------------!
!use 1 ghost cells within the domain (1 points overlap between domains)

!Allocate local arrays
allocate(u(0:nx+1,0:ny+1)) 	!primary array for conservative field variables
allocate(u1(0:nx+1,0:ny+1)) !Time integration substage array
allocate(s(0:nx+1,0:ny+1)) 	!Time integration substage array

do j =1, ny
do i =1, nx
    u(i,j) = 0.0d0
end do
end do

!initial conditions for heat equation
call initialize(nx,ny,x,y,u)

!Calculate timestep
dt = cfl*dx*dx/(alpha)

if (myid.eq.0) then
print*,dt
print*,cfl
print*,dx
print*,alpha
end if

    
!Time integration
!TVDRK3 coefficient
a = 1.0d0/3.0d0
b = 2.0d0/3.0d0
snap_time = 0.0d0
snapshot = 0
t = 0.0d0

do while (t<tmax)
t = t+dt

if (t>snap_time) then
!----------------------------------------------------------------------!
! Wait until all processors come to this point before writing files
!----------------------------------------------------------------------!
call MPI_Barrier(MPI_COMM_WORLD, ierr)
!First plot output
call output(nx,ny,x,y,u,myid,snapshot)
snap_time = snap_time + tmax/dfloat(nsnap)
snapshot = snapshot + 1
end if

	call rhs(nx,ny,dx,dy,u,s)
    
    !--------------------!
	!Step 1
    !--------------------!
    do j=1,ny
	do i=1,nx
    u1(i,j) = u(i,j) + dt*s(i,j) 
    end do
    end do

    !update boundary conditions only x and y (global)
    call bc_xy(nx,ny,u1)    
              
    !--------------------!
	!Step 2
    !--------------------!
	call rhs(nx,ny,dx,dy,u1,s)
   
    do j=1,ny
	do i=1,nx
    u1(i,j) = 0.75d0*u(i,j) + 0.25d0*u1(i,j) + 0.25d0*dt*s(i,j) 
    end do
    end do

    !update boundary conditions
    call bc_xy(nx,ny,u1)
        
    !--------------------!
	!Step 3 
    !--------------------! 

	call rhs(nx,ny,dx,dy,u1,s)

    do j=1,ny
	do i=1,nx
    u(i,j) = a*u(i,j) + b*u1(i,j) + b*dt*s(i,j) 
    end do
    end do
    
    !update boundary conditions
    call bc_xy(nx,ny,u)

end do

!----------------------------------------------------------------------!
! MPI final call
!----------------------------------------------------------------------!
call MPI_Finalize(ierr) 

end program


!-----------------------------------------------------------------------------------!
!Initial conditions and problem definition
!-----------------------------------------------------------------------------------!
subroutine initialize(nx,ny,x,y,u)
implicit none
integer::nx,ny,i,j!,numthreads,omp_get_num_threads,myid
real*8 ::u(0:nx+1,0:ny+1),x(0:nx+1),y(0:ny+1)

!$omp parallel private (i,j) shared (x,y,u,nx,ny)
!$omp do
do j = 1,ny
  do i = 1,nx
    u(i,j) = dsin(x(i))+dcos(y(j))
  end do
end do
!$omp end do
!$omp end parallel

!print*,numthreads

call bc_xy(nx,ny,u)

return
end


!-----------------------------------------------------------------------------------!
!Boundary conditions 
!Periodic in y and x
!-----------------------------------------------------------------------------------!
subroutine bc_xy(nx,ny,u)
implicit none
include 'mpif.h'	!MPI header file

integer::nx,ny,i,j,ic,myid,nsend,ierr,np
real*8 ::u(0:nx+1,0:ny+1)

integer,dimension(4) :: req = MPI_REQUEST_NULL
integer               :: status(MPI_STATUS_SIZE)
integer               :: status_array(MPI_STATUS_SIZE,4) !total 4 MPI requests
integer, parameter    :: id_top2bottom_1 = 1000 !message tag 
integer, parameter    :: id_bottom2top_1 = 1001 !message tag


common /mpivars/myid,nsend,np


!No slip BCs for x direction (no MPI in this direction)

!$omp parallel private (i,j) shared (u,nx,ny)
!$omp do
do j=1,ny
	u(0,j) =-u(1,j)		!left-wall
	u(nx+1,j) =-u(nx,j)		!right-wall
end do
!$omp end do
!$omp end parallel


!SEND/RECEIVE DATA AMONG BOUNDARY PROCESSORS for y direction
        
        !MPI based BCs for y direction
        ic = 0
     	!send buffer from top part of the global boundary
     	if (myid .eq. np-1) then    
        	ic = ic + 1
     		call MPI_Isend(u(0,ny), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
     	end if

	 	!send buffer from bottom part of the global boundary
     	if (myid .eq. 0) then
        	ic = ic + 1
       		call MPI_Isend(u(0,1), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
     	end if

     	!receive buffer for bottom part of the global boundary 
     	if (myid .eq. 0) then
        	ic = ic + 1
        	call MPI_Irecv(u(0,0), nsend, MPI_DOUBLE_PRECISION, &
                       np-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
     	end if

	 	!receive buffer for top part of the global boundary 
     	if (myid .eq. np-1) then
        	ic = ic + 1
       		call MPI_Irecv(u(0,ny+1), nsend, MPI_DOUBLE_PRECISION, &
                       0, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
     	end if

	 	! To wait until Isend and Irecv in the above are completed
     	call MPI_Waitall(ic, req, status_array, ierr)

      
    !SEND/RECEIVE DATA AMONG PROCESSORS
	ic = 0
    !send buffer from top part of each local zone
    if (myid .ne. np-1) then
        ic = ic + 1
     	call MPI_Isend(u(0,ny), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
    end if

	!send buffer from bottom part of each local zone
    if (myid .ne. 0) then
        ic = ic + 1
       	call MPI_Isend(u(0,1), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
    end if

    !receive buffer for bottom part of each local zone 
    if (myid .ne. 0) then
        ic = ic + 1
        call MPI_Irecv(u(0,0), nsend, MPI_DOUBLE_PRECISION, &
                       myid-1, id_top2bottom_1, MPI_COMM_WORLD, req(ic), ierr)
    end if

	!receive buffer for top part of each local zone
    if (myid .ne. np-1) then
        ic = ic + 1
       	call MPI_Irecv(u(0,ny+1), nsend, MPI_DOUBLE_PRECISION, &
                       myid+1, id_bottom2top_1, MPI_COMM_WORLD, req(ic), ierr)
    end if

	! To wait until Isend and Irecv in the above are completed
    call MPI_Waitall(ic, req, status_array, ierr)

return
end

!----------------------------------------------------------------------!
! Output files: ASCII - Tecplot format
!----------------------------------------------------------------------!
subroutine output(nx,ny,x,y,u,myid,snapshot)
implicit none
integer::nx,ny,myid,snapshot
real*8 ::u(0:nx+1,0:ny+1)
real*8 ::x(0:nx+1)
real*8 ::y(0:ny+1)
integer:: i,j
character(80):: charID, snapID, filename

write(charID,'(i5)') myid       !index for each processor 
write(snapID,'(i5)') snapshot      !index for time snapshot

filename = 'data_'// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.dat'
! Define the file name (write all conserved variables)
open(unit=19, file=filename)
! Tecplot header
write(19,*) 'title =', '"Zone_'// trim(adjustl(snapID)) //'_'// trim(adjustl(charID))//'"'
write(19,*) 'variables = "x", "y","u"'

! Open the file and start writing the data
do j = 0, ny
do i = 0, nx
  	write(19,*) x(i),y(j),u(i,j)
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



!----------------------------------------------------------------------!
!Subroutine for heat equation RHS
!----------------------------------------------------------------------!
subroutine rhs(nx,ny,dx,dy,u,s)
implicit none

integer :: nx,ny,i,j
real*8 :: dx, dy, alpha, term1, term2
real*8,dimension(0:nx+1,0:ny+1) :: u,s

common /heatcoeff/ alpha

!$omp parallel private (i,j,term1,term2) shared (u,alpha,nx,ny)
!$omp do
do j = 1,ny
  do i = 1,nx
    term1 = (u(i+1,j)+u(i-1,j)-2.0d0*u(i,j))/(dx*dx)
    term2 = (u(i,j+1)+u(i,j-1)-2.0d0*u(i,j))/(dy*dy)
    s(i,j) = alpha*(term1 + term2)
  end do
end do
!$omp end do
!$omp end parallel

return
end
