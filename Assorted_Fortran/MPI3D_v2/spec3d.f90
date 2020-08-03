!------------------------------------------------------------------------------!
!Compute energy spectrum from saved data produced by euler3d_MPI.f90
!------------------------------------------------------------------------------!
!Omer San
!Oklahoma State University, Stillwater
!CFDLab.org, cfdlab.osu@gmail.com
!Updated: May 15, 2017
!------------------------------------------------------------------------------!

program spec3d
implicit none
integer, parameter::nsnap_s=1 	!snapshot id in order to process
integer, parameter::nsnap_e=5 	!snapshot id in order to process
integer, parameter::np=96     	!number of processor (nodes x ppn)
integer :: nx,ny,nz   			!global number of data point in x (even, resolution)
real*8,dimension(:,:,:,:),allocatable :: q
real*8,dimension(:,:,:),allocatable:: u,v,w
integer:: i,j,k,myid,kk,ifile,myid2,nzl,iopt
character(80):: charID,snapID,filename,fi2
real*8 :: dx,dy,dz,x0,y0,z0,lx,ly,lz,pi,x,y,z


nx = 512
ny = nx
nz = nx

iopt = 1 ![1]cube 2pi, [2]cube 1 unit
if (iopt.eq.1) then
pi = 4.0d0*datan(1.0d0)
dx = 2.0d0*pi/dfloat(nx)
dy = dx
dz = dx

x0 = 0.0d0
y0 = 0.0d0
z0 = 0.0d0
else
lx = 1.0d0
ly = 1.0d0
lz = 1.0d0

dx = 1.0d0/dfloat(nx)
dy = dx
dz = dx

x0 =-lx/2.0d0
y0 =-ly/2.0d0
z0 =-lz/2.0d0
end if

!Allocate conserved variables
allocate(q(0:nx,0:ny,0:nz,5))

!allocate velocity components for postprocessing
allocate(u(0:nx,0:ny,0:nz))
allocate(v(0:nx,0:ny,0:nz))
allocate(w(0:nx,0:ny,0:nz))

do ifile=nsnap_s,nsnap_e

kk = 0

do myid = 0,np-1
  
write(charID,'(i5)') myid       !index for each processor 
write(snapID,'(i5)') ifile      !index for time snapshot


! Read data file
! Read data file
fi2= "load_"// trim(adjustl(charID)) // '.dat'
open(unit=19, file=fi2)
read(19,*)myid2,nzl
close(19)


! Define the file name
filename = "data_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.dat'

! Open the file and start reading the data
open(unit=19, file=filename)
do k = 0, nzl
do j = 0, ny
do i = 0, nx
  	read(19,*) q(i,j,k+kk,1), q(i,j,k+kk,2),q(i,j,k+kk,3),q(i,j,k+kk,4),q(i,j,k+kk,5)
end do
end do
end do

close(19)

kk = kk + nzl


end do


! writing density field at cell centers:
! Define the file name
filename = "a_density_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z", "r"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', nx+1, ' j=', ny+1,' k=', nz+1, ' f=point'
do k=0,nz            
do j=0,ny
do i=0,nx
	x = x0 - 0.5d0*dx +  dfloat(i)*dx 
    y = y0 - 0.5d0*dy +  dfloat(j)*dy 
    z = z0 - 0.5d0*dz +  dfloat(k)*dz   
write(19,*)x,y,z,q(i,j,k,1)
end do
end do
end do
close(19)


!compute velocity components
do k=0,nz
do j=0,ny
do i=0,nx
  	u(i,j,k)= q(i,j,k,2)/q(i,j,k,1)
    v(i,j,k)= q(i,j,k,3)/q(i,j,k,1)
    w(i,j,k)= q(i,j,k,4)/q(i,j,k,1)
end do
end do
end do

! writing density field at cell centers:
! Define the file name
filename = "a_vel_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z", "u", "v", "w"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', nx+1, ' j=', ny+1,' k=', nz+1, ' f=point'
do k=0,nz            
do j=0,ny
do i=0,nx
    x = x0 - 0.5d0*dx +  dfloat(i)*dx 
    y = y0 - 0.5d0*dy +  dfloat(j)*dy 
    z = z0 - 0.5d0*dz +  dfloat(k)*dz   
write(19,*)x,y,z,u(i,j,k),v(i,j,k),w(i,j,k)
end do
end do
end do
close(19)


!compute vorticity and Qcriterion field:
!here 6th order compact scheme is used to compute derivatives 
call Qcriterion(nx,ny,nz,dx,dy,dz,x0,y0,z0,u,v,w,ifile)

!compute energy spectrum (Kida):
!fast Fourier transform is used
call spectrum3d(nx,ny,nz,u,v,w,ifile)

!compute structure functions:
!both longitudinal and transverse components are computed for second and third order SF
call strfun3d(nx,ny,nz,dy,u,v,w,ifile)

end do

end


!-----------------------------------------------------------------!
!Compute structure functions from 3D velocity field by definition 
!Boffetta, Musacchio, PRE 2010
!Kramer et al., PRE 2011
!-----------------------------------------------------------------!
subroutine strfun3d(nx,ny,nz,dh,u,v,w,ifile)
implicit none
integer::nx,ny,nz,ifile
real*8 ::u(0:nx,0:ny,0:nz)
real*8 ::v(0:nx,0:ny,0:nz)
real*8 ::w(0:nx,0:ny,0:nz)
integer::i,j,k,ii,jj,kk,ir,jr,kr
real*8 ::dh,dl2(1:ny/2),dt2(1:ny/2),dl3(1:ny/2),dt3(1:ny/2)
character(80):: snapID,filename


write(snapID,'(i5)') ifile      !index for time snapshot


do ii=1,nx/2
 
    dl2(ii) = 0.0d0
    dt2(ii) = 0.0d0

    dl3(ii) = 0.0d0
    dt3(ii) = 0.0d0
         	
	do k=1,nz
	do j=1,ny
	do i=1,nx
    
    	if (i+ii.ge.nx) then
        ir = ii-nx
        else
        ir = ii
        end if
        
		dl2(ii)=dl2(ii)+(dabs(u(i+ir,j,k)-u(i,j,k)))**2 !longitudinal
        dt2(ii)=dt2(ii)+(dabs(v(i+ir,j,k)-v(i,j,k)))**2 !transverse
        dt2(ii)=dt2(ii)+(dabs(w(i+ir,j,k)-w(i,j,k)))**2 !transverse
            
		dl3(ii)=dl3(ii)+(dabs(u(i+ir,j,k)-u(i,j,k)))**3 !longitudinal
        dt3(ii)=dt3(ii)+(dabs(v(i+ir,j,k)-v(i,j,k)))**3 !transverse
        dt3(ii)=dt3(ii)+(dabs(w(i+ir,j,k)-w(i,j,k)))**3 !transverse
        
	end do
	end do
    end do

end do

do jj=1,ny/2
 
    do k=1,nz
	do j=1,ny
	do i=1,nx
    
    	if (j+jj.ge.ny) then
        jr = jj-ny
        else
        jr = jj
        end if
        
		dl2(jj)=dl2(jj)+(dabs(v(i,j+jr,k)-v(i,j,k)))**2 !longitudinal
        dt2(jj)=dt2(jj)+(dabs(u(i,j+jr,k)-u(i,j,k)))**2 !transverse
        dt2(jj)=dt2(jj)+(dabs(w(i,j+jr,k)-w(i,j,k)))**2 !transverse
 
		dl3(jj)=dl3(jj)+(dabs(v(i,j+jr,k)-v(i,j,k)))**3 !longitudinal
        dt3(jj)=dt3(jj)+(dabs(u(i,j+jr,k)-u(i,j,k)))**3 !transverse
        dt3(jj)=dt3(jj)+(dabs(w(i,j+jr,k)-w(i,j,k)))**3 !transverse
                   
	end do
	end do
    end do


end do


do kk=1,nz/2
 
    do k=1,nz
	do j=1,ny
	do i=1,nx
    
    	if (k+kk.ge.nz) then
        kr = kk-nz
        else
        kr = kk
        end if
        
		dl2(kk)=dl2(kk)+(dabs(w(i,j,k+kr)-w(i,j,k)))**2 !longitudinal
        dt2(kk)=dt2(kk)+(dabs(u(i,j,k+kr)-u(i,j,k)))**2 !transverse
        dt2(kk)=dt2(kk)+(dabs(v(i,j,k+kr)-v(i,j,k)))**2 !transverse
 
		dl3(kk)=dl3(kk)+(dabs(w(i,j,k+kr)-w(i,j,k)))**3 !longitudinal
        dt3(kk)=dt3(kk)+(dabs(u(i,j,k+kr)-u(i,j,k)))**3 !transverse
        dt3(kk)=dt3(kk)+(dabs(v(i,j,k+kr)-v(i,j,k)))**3 !transverse
                   
	end do
	end do
    end do


end do


do jj=1,ny/2     

        dl2(jj) = dl2(jj)/dfloat(3*nx*ny*nz)
        dt2(jj) = dt2(jj)/dfloat(6*nx*ny*nz)

        dl3(jj) = dl3(jj)/dfloat(3*nx*ny*nz)
        dt3(jj) = dt3(jj)/dfloat(6*nx*ny*nz)

end do
        
  

! Define the file name
filename = "a_str_long_2nd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dh*dfloat(jj),dl2(jj)
end do
close(802)

! Define the file name
filename = "a_str_tran_2nd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dh*dfloat(jj),dt2(jj)
end do
close(802)

! Define the file name
filename = "a_str_long_3rd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dh*dfloat(jj),dl3(jj)
end do
close(802)

! Define the file name
filename = "a_str_tran_3rd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dh*dfloat(jj),dt3(jj)
end do
close(802)



return
end 

!-----------------------------------------------------------------!
!Q-criterion
!-----------------------------------------------------------------!
subroutine Qcriterion(nx,ny,nz,dx,dy,dz,x0,y0,z0,u,v,w,ifile)
implicit none
integer::nx,ny,nz,ifile
real*8 ::dx,dy,dz,x0,y0,z0,x,y,z
real*8 ::u(0:nx,0:ny,0:nz)
real*8 ::v(0:nx,0:ny,0:nz)
real*8 ::w(0:nx,0:ny,0:nz)
integer::i,j,k
real*8, dimension (:), allocatable :: aa,bb
real*8,dimension(:,:,:),allocatable:: qc,wa,w1,w2,w3,ux,vx,wx,uy,vy,wy,uz,vz,wz
character(80):: snapID,filename


write(snapID,'(i5)') ifile      !index for time snapshot


allocate(qc(0:nx,0:ny,0:nz))
allocate(w1(0:nx,0:ny,0:nz))
allocate(w2(0:nx,0:ny,0:nz))
allocate(w3(0:nx,0:ny,0:nz))
allocate(wa(0:nx,0:ny,0:nz))

allocate(ux(0:nx,0:ny,0:nz))
allocate(uy(0:nx,0:ny,0:nz))
allocate(uz(0:nx,0:ny,0:nz))
allocate(vx(0:nx,0:ny,0:nz))
allocate(vy(0:nx,0:ny,0:nz))
allocate(vz(0:nx,0:ny,0:nz))
allocate(wx(0:nx,0:ny,0:nz))
allocate(wy(0:nx,0:ny,0:nz))
allocate(wz(0:nx,0:ny,0:nz))

! u_x
allocate(aa(0:nx),bb(0:nx))
do k=0,nz
do j=0,ny
	do i=0,nx
	aa(i) = u(i,j,k)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	ux(i,j,k) = bb(i)
	end do
end do
end do
deallocate(aa,bb)

! u_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
do k=0,nz
	do j=0,ny
	aa(j) = u(i,j,k)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	uy(i,j,k) = bb(j)
	end do
end do
end do
deallocate(aa,bb)

! u_z
allocate(aa(0:nz),bb(0:nz))
do j=0,ny
do i=0,nx
	do k=0,nz
	aa(k) = u(i,j,k)
	end do
		call c6dp(aa,bb,dz,nz)
	do k=0,nz
	uz(i,j,k) = bb(k)
	end do
end do
end do
deallocate(aa,bb)


! v_x
allocate(aa(0:nx),bb(0:nx))
do k=0,nz
do j=0,ny
	do i=0,nx
	aa(i) = v(i,j,k)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	vx(i,j,k) = bb(i)
	end do
end do
end do
deallocate(aa,bb)

! v_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
do k=0,nz
	do j=0,ny
	aa(j) = v(i,j,k)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	vy(i,j,k) = bb(j)
	end do
end do
end do
deallocate(aa,bb)

! v_z
allocate(aa(0:nz),bb(0:nz))
do j=0,ny
do i=0,nx
	do k=0,nz
	aa(k) = v(i,j,k)
	end do
		call c6dp(aa,bb,dz,nz)
	do k=0,nz
	vz(i,j,k) = bb(k)
	end do
end do
end do
deallocate(aa,bb)


! w_x
allocate(aa(0:nx),bb(0:nx))
do k=0,nz
do j=0,ny
	do i=0,nx
	aa(i) = w(i,j,k)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	wx(i,j,k) = bb(i)
	end do
end do
end do
deallocate(aa,bb)

! w_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
do k=0,nz
	do j=0,ny
	aa(j) = w(i,j,k)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	wy(i,j,k) = bb(j)
	end do
end do
end do
deallocate(aa,bb)

! w_z
allocate(aa(0:nz),bb(0:nz))
do j=0,ny
do i=0,nx
	do k=0,nz
	aa(k) = w(i,j,k)
	end do
		call c6dp(aa,bb,dz,nz)
	do k=0,nz
	wz(i,j,k) = bb(k)
	end do
end do
end do
deallocate(aa,bb)


!Compute vorticity
!Q-criterion:qc
!Absolute value of vorticity:wa
do k=0,nz
do j=0,ny
do i=0,nx

  	w1(i,j,k)= wy(i,j,k) - vz(i,j,k)
    w2(i,j,k)= uz(i,j,k) - wx(i,j,k)
    w3(i,j,k)= vx(i,j,k) - uy(i,j,k)

	qc(i,j,k) = -0.5d0*(ux(i,j,k)*ux(i,j,k)+uy(i,j,k)*vx(i,j,k)+uz(i,j,k)*wx(i,j,k) &
                      +vx(i,j,k)*uy(i,j,k)+vy(i,j,k)*vy(i,j,k)+vz(i,j,k)*wy(i,j,k) &
                      +wx(i,j,k)*uz(i,j,k)+wy(i,j,k)*vz(i,j,k)+wz(i,j,k)*wz(i,j,k) )
	
    wa(i,j,k)=dsqrt(w1(i,j,k)*w1(i,j,k)+w2(i,j,k)*w2(i,j,k)+w3(i,j,k)*w3(i,j,k))
end do
end do
end do


! writing vorticity field at cell centers:
! Define the file name
filename = "a_vorticity_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z", "w1","w2","w3"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', nx+1, ' j=', ny+1,' k=', nz+1, ' f=point'
do k=0,nz            
do j=0,ny
do i=0,nx
  	x = x0 - 0.5d0*dx +  dfloat(i)*dx 
    y = y0 - 0.5d0*dy +  dfloat(j)*dy 
    z = z0 - 0.5d0*dz +  dfloat(k)*dz   
write(19,*)x,y,z,w1(i,j,k),w2(i,j,k),w3(i,j,k)
end do
end do
end do
close(19)


! Define the file name
filename = "a_Qcriterion_"// trim(adjustl(snapID)) //'.plt'
 
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z", "Q","w"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', nx+1, ' j=', ny+1,' k=', nz+1, ' f=point'
do k=0,nz            
do j=0,ny
do i=0,nx
  	x = x0 - 0.5d0*dx +  dfloat(i)*dx 
    y = y0 - 0.5d0*dy +  dfloat(j)*dy 
    z = z0 - 0.5d0*dz +  dfloat(k)*dz   
write(19,*)x,y,z,qc(i,j,k),wa(i,j,k)
end do
end do
end do
close(19)


deallocate(qc,wa,w1,w2,w3,ux,vx,wx,uy,vy,wy,uz,vz,wz)


return 
end 

!-----------------------------------------------------------------!
!Compute vorticity using compact scheme
!-----------------------------------------------------------------!
subroutine vorticity3d(nx,ny,nz,dx,dy,dz,x0,y0,z0,q,ifile)
implicit none
integer::nx,ny,nz,ifile,i,j,k
real*8 ::dx,dy,dz,x0,y0,z0,x,y,z
real*8 ::q(0:nx,0:ny,0:nz,5)
real*8,dimension(:,:,:),allocatable:: w1,w2,w3
real*8,dimension(:), allocatable:: aa,bb
character(80):: snapID,filename

write(snapID,'(i5)') ifile      !index for time snapshot

allocate(w1(0:nx,0:ny,0:nz))
allocate(w2(0:nx,0:ny,0:nz))
allocate(w3(0:nx,0:ny,0:nz))


! compute vorticity w3
! v_x
allocate(aa(0:nx),bb(0:nx))
do k=0,nz
do j=0,ny
	do i=0,nx
	aa(i) = q(i,j,k,3)/q(i,j,k,1)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	w3(i,j,k) = bb(i)
	end do
end do
end do
deallocate(aa,bb)

! u_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
  do k=0,nz
	do j=0,ny
	aa(j) = q(i,j,k,2)/q(i,j,k,1)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	w3(i,j,k) = w3(i,j,k) - bb(j)
	end do
end do
end do
deallocate(aa,bb)


! compute vorticity w2
! u_z
allocate(aa(0:nz),bb(0:nz))
do j=0,ny
do i=0,nx
	do k=0,nz
	aa(k) = q(i,j,k,2)/q(i,j,k,1)
	end do
		call c6dp(aa,bb,dz,nz)
	do k=0,nz
	w2(i,j,k) = bb(k)
	end do
end do
end do
deallocate(aa,bb)

! w_x
allocate(aa(0:nx),bb(0:nx))
do k=0,nz
do j=0,ny
	do i=0,nx
	aa(i) = q(i,j,k,4)/q(i,j,k,1)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	w2(i,j,k) = w2(i,j,k) - bb(i)
	end do
end do
end do
deallocate(aa,bb)

! compute vorticity w1
! w_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
do k=0,nz
	do j=0,ny
	aa(j) = q(i,j,k,4)/q(i,j,k,1)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	w1(i,j,k) = bb(j)
	end do
end do
end do
deallocate(aa,bb)

! v_z
allocate(aa(0:nz),bb(0:nz))
do j=0,ny
do i=0,nx
	do k=0,nz
	aa(k) = q(i,j,k,3)/q(i,j,k,1)
	end do
		call c6dp(aa,bb,dz,nz)
	do k=0,nz
	w1(i,j,k) = w1(i,j,k)-bb(k)
	end do
end do
end do
deallocate(aa,bb)

! writing vorticity field at cell centers:
! Define the file name
filename = "a_vorticity_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z", "w1","w2","w3"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', nx+1, ' j=', ny+1,' k=', nz+1, ' f=point'
do k=0,nz            
do j=0,ny
do i=0,nx
	x = x0 - 0.5d0*dx +  dfloat(i)*dx 
    y = y0 - 0.5d0*dy +  dfloat(j)*dy 
    z = z0 - 0.5d0*dz +  dfloat(k)*dz   
write(19,*)x,y,z,w1(i,j,k),w2(i,j,k),w3(i,j,k)
end do
end do
end do
close(19)

deallocate(w1,w2,w3)

return
end

!-----------------------------------------------------------------!
!Compute spectrum from by definition of Kida
!S Kida, Y. Murakami, et al. 1990
!-----------------------------------------------------------------!
subroutine spectrum3d(nx,ny,nz,u,v,w,ifile)
implicit none
integer::nx,ny,nz,ifile
real*8 ::u(0:nx,0:ny,0:nz)
real*8 ::v(0:nx,0:ny,0:nz)
real*8 ::w(0:nx,0:ny,0:nz)
integer::i,j,k,p,np
real*8 ::kx(0:nx),ky(0:ny),kz(0:nz)
real*8,dimension(:),allocatable:: data1d,data2d,data3d,ae
real*8,dimension(:,:,:),allocatable::ee
integer,parameter::ndim=3
integer::nn(ndim),isign
real*8 ::temp,kr
character(80):: snapID,filename

write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny*nz))
allocate(data2d(2*nx*ny*nz))
allocate(data3d(2*nx*ny*nz))

nn(1)= nx
nn(2)= ny
nn(3)= nz

!wave numbers (sequence is important)
p=0
do i=0,nx/2
kx(i) = dfloat(p)
p=p+1
end do
p=-nx/2+1
do i=nx/2+1,nx-1
kx(i) = dfloat(p)
p=p+1
end do

p=0
do j=0,ny/2
ky(j) = dfloat(p)
p=p+1
end do
p=-ny/2+1
do j=ny/2+1,ny-1
ky(j) = dfloat(p)
p=p+1
end do

p=0
do k=0,nz/2
kz(k) = dfloat(p)
p=p+1
end do
p=-nz/2+1
do k=nz/2+1,nz-1
kz(k) = dfloat(p)
p=p+1
end do

!finding fourier coefficients of u and v
!invese fourier transform
p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j,k)
data1d(p+1)=0.0d0
data2d(p)  =v(i,j,k)
data2d(p+1)=0.0d0
data3d(p)  =w(i,j,k)
data3d(p+1)=0.0d0
p=p+2
end do
end do
end do

isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)
call fourn(data3d,nn,ndim,isign)

temp=1.0d0/dfloat(nx*ny*nz)

!normalize
do p=1,2*nx*ny*nz
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
data3d(p)=data3d(p)*temp
end do

!Energy density
allocate(ee(0:nx-1,0:ny-1,0:nz-1))

p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
ee(i,j,k)=0.5d0*(data1d(p)*data1d(p) + data1d(p+1)*data1d(p+1) &
                +data2d(p)*data2d(p) + data2d(p+1)*data2d(p+1) &
                +data3d(p)*data3d(p) + data3d(p+1)*data3d(p+1) )
p=p+2
end do
end do
end do

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2 + (kz(nz/2))**2)) 

allocate(ae(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
    do k=0,nz-1
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j,k)
		end if
	end do
    end do
    end do
end do

! Define the file name
filename = "a_spectrum_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","E(k)"'
do p=1,np
write(802,*)dfloat(p),ae(p)
end do
close(802)

open(15,file="a_spectrum_k53.plt")
write(15,*) 'variables ="k","E(k)"'
write(15,*)2.0d0, 1.0d1
write(15,*)2.0d2, 1.0d1*1.0d2**(-5.0d0/3.0d0)
close(15)

deallocate(data1d,data2d,ee,ae)

return
end 


!-----------------------------------------------------------------!
!Compute spectrum from by definition of Kida
!S Kida, Y. Murakami, et al. 1990
!-----------------------------------------------------------------!
subroutine spectrum3d_old(nx,ny,nz,q,ifile)
implicit none
integer::nx,ny,nz,ifile
real*8 ::q(0:nx,0:ny,0:nz,5)
integer::i,j,k,p,np
real*8 ::kx(0:nx),ky(0:ny),kz(0:nz)
real*8,dimension(:),allocatable:: data1d,data2d,data3d,ae
real*8,dimension(:,:,:),allocatable::ee
integer,parameter::ndim=3
integer::nn(ndim),isign
real*8 ::temp,kr
character(80):: snapID,filename

write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny*nz))
allocate(data2d(2*nx*ny*nz))
allocate(data3d(2*nx*ny*nz))

nn(1)= nx
nn(2)= ny
nn(3)= nz

!wave numbers (sequence is important)
p=0
do i=0,nx/2
kx(i) = dfloat(p)
p=p+1
end do
p=-nx/2+1
do i=nx/2+1,nx-1
kx(i) = dfloat(p)
p=p+1
end do

p=0
do j=0,ny/2
ky(j) = dfloat(p)
p=p+1
end do
p=-ny/2+1
do j=ny/2+1,ny-1
ky(j) = dfloat(p)
p=p+1
end do

p=0
do k=0,nz/2
kz(k) = dfloat(p)
p=p+1
end do
p=-nz/2+1
do k=nz/2+1,nz-1
kz(k) = dfloat(p)
p=p+1
end do

!finding fourier coefficients of u and v
!invese fourier transform
p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =q(i,j,k,2)/q(i,j,k,1)
data1d(p+1)=0.0d0
data2d(p)  =q(i,j,k,3)/q(i,j,k,1)
data2d(p+1)=0.0d0
data3d(p)  =q(i,j,k,4)/q(i,j,k,1)
data3d(p+1)=0.0d0
p=p+2
end do
end do
end do

isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)
call fourn(data3d,nn,ndim,isign)

temp=1.0d0/dfloat(nx*ny*nz)

!normalize
do p=1,2*nx*ny*nz
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
data3d(p)=data3d(p)*temp
end do

!Energy density
allocate(ee(0:nx-1,0:ny-1,0:nz-1))

p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
ee(i,j,k)=0.5d0*(data1d(p)*data1d(p) + data1d(p+1)*data1d(p+1) &
                +data2d(p)*data2d(p) + data2d(p+1)*data2d(p+1) &
                +data3d(p)*data3d(p) + data3d(p+1)*data3d(p+1) )
p=p+2
end do
end do
end do

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2 + (kz(nz/2))**2)) 

allocate(ae(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
    do k=0,nz-1
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j,k)
		end if
	end do
    end do
    end do
end do

! Define the file name
filename = "a_spectrum_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","E(k)"'
do p=1,np
write(802,*)dfloat(p),ae(p)
end do
close(802)

open(15,file="a_spectrum_k53.plt")
write(15,*) 'variables ="k","E(k)"'
write(15,*)2.0d0, 1.0d1
write(15,*)2.0d2, 1.0d1*1.0d2**(-5.0d0/3.0d0)
close(15)

deallocate(data1d,data2d,ee,ae)

return
end 


!-----------------------------------------------------------------!
!Compute spectrum by definition of Kida (avaraged version)
!S Kida, Y. Murakami, et al. 1990
!-----------------------------------------------------------------!
subroutine spectrum_ave(nx,ny,nz,q,ifile)
implicit none
integer::nx,ny,nz,i,j,k,p,nt,ic1,ifile
real*8 ::temp,kk,pi
real*8 ::q(0:nx,0:ny,0:nz,5)
integer,parameter:: ndim=3
integer::nn(ndim),isign
real*8,parameter:: tiny=1.0d-10
integer::kx(0:nx),ky(0:ny),kz(0:nz)
complex*16,dimension(:,:,:),allocatable::uf,vf,wf
real*8,dimension(:,:,:),allocatable::es
real*8,dimension(:),allocatable::aes1
real*8,dimension(:),allocatable ::data1d
character(80):: snapID,filename



write(snapID,'(i5)') ifile      !index for time snapshot


nn(1)= nx
nn(2)= ny
nn(3)= nz

temp=1.0d0/dfloat(nx*ny*nz)

pi=4.0d0*datan(1.0d0)

!wave numbers (sequence is important)
p=0
do i=0,nx/2
kx(i) = dfloat(p)
p=p+1
end do
p=-nx/2+1
do i=nx/2+1,nx-1
kx(i) = dfloat(p)
p=p+1
end do

p=0
do j=0,ny/2
ky(j) = dfloat(p)
p=p+1
end do
p=-ny/2+1
do j=ny/2+1,ny-1
ky(j) = dfloat(p)
p=p+1
end do

p=0
do k=0,nz/2
kz(k) = dfloat(p)
p=p+1
end do
p=-nz/2+1
do k=nz/2+1,nz-1
kz(k) = dfloat(p)
p=p+1
end do


allocate(data1d(2*nx*ny*nz))
allocate(uf(0:nx,0:ny,0:nz))
allocate(vf(0:nx,0:ny,0:nz))
allocate(wf(0:nx,0:ny,0:nz))

!finding fourier coefficients of u 
!inverse fourier transform
p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =q(i,j,k,2)/q(i,j,k,1)
data1d(p+1)=0.0d0
p=p+2
end do
end do
end do

isign=-1
call fourn(data1d,nn,ndim,isign)

p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
uf(i,j,k)=dcmplx(data1d(p)*temp,data1d(p+1)*temp)
p=p+2
end do
end do
end do

!finding fourier coefficients of v 
!invese fourier transform
p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =q(i,j,k,3)/q(i,j,k,1)
data1d(p+1)=0.0d0
p=p+2
end do
end do
end do

isign=-1
call fourn(data1d,nn,ndim,isign)

p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
vf(i,j,k)=dcmplx(data1d(p)*temp,data1d(p+1)*temp)
p=p+2
end do
end do
end do


!finding fourier coefficients of w 
!invese fourier transform
p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =q(i,j,k,4)/q(i,j,k,1)
data1d(p+1)=0.0d0
p=p+2
end do
end do
end do

isign=-1
call fourn(data1d,nn,ndim,isign)

p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
wf(i,j,k)=dcmplx(data1d(p)*temp,data1d(p+1)*temp)
p=p+2
end do
end do
end do

!Compute energy spectrum:es

allocate(es(0:nx,0:ny,0:nz))

do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
   
es(i,j,k)=0.5d0*dreal(dconjg(uf(i,j,k))*uf(i,j,k) &
                     +dconjg(vf(i,j,k))*vf(i,j,k) &
			         +dconjg(wf(i,j,k))*wf(i,j,k) ) 

end do
end do
end do

nt = nint(dsqrt(dfloat((kx(nx/2))**2 + (ky(ny/2))**2 + (kz(nz/2))**2))) 

allocate(aes1(nt))

!averaging
do p=1,nt
	aes1(p)=0.0d0
	ic1=0
	do k=0,nz-1
	do j=0,ny-1
	do i=0,nx-1
		kk=dsqrt(dfloat(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k)))
		if (kk.ge.(dfloat(p)-0.5d0).and.kk.lt.(dfloat(p)+0.5d0)) then
		ic1=ic1+1
		aes1(p)=aes1(p)+es(i,j,k)
		end if
	end do 
	end do
	end do
	aes1(p)=aes1(p)/(dfloat(ic1)+tiny)*(4.0d0*pi*(dfloat(p)*dfloat(p)))
end do


! Define the file name
filename = "a_spectrum_averaged_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=800, file=filename)
write(800,*)'variables ="k","E(k)"'
do p=1,nt
write(800,*)p,aes1(p) 
end do
close(800)


deallocate(uf,vf,wf,es,aes1,data1d)

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
