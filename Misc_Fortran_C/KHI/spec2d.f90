!------------------------------------------------------------------------------!
!Compute energy spectrum from computed data produced by euler2d_MPI.f90
!To run: gfortran -o aa spec2d.f90
!------------------------------------------------------------------------------!
!Omer San
!Oklahoma State University, Stillwater
!CFDLab.org, cfdlab.osu@gmail.com
!Updated: June 18, 2016
!------------------------------------------------------------------------------!

program spec2d
implicit none
integer, parameter::nsnap_s=1 !snapshot id in order to process
integer, parameter::nsnap_e=5 !snapshot id in order to process
integer, parameter::np=12  !number of processor (nodes x ppn)
integer :: nx,ny  !global number of data point in x (even, resolution)
!integer, parameter::ny=nx  	!global number of data point in y (local ny=ny/np)
!real*8 ::u(0:nx,0:ny) 
!real*8 ::v(0:nx,0:ny) 
!real*8 ::rh(0:nx,0:ny) 
!real*8 ::w(0:nx,0:ny) 
!real*8 ::x(0:nx,0:ny) 
!real*8 ::y(0:nx,0:ny) 
real*8,dimension(:,:),allocatable :: u,v,w,rh,p,x,y
integer:: i,j,myid,jj,ifile,myid2,nyl,ipr
character(80):: charID,snapID,filename,fi2
real*8 :: dx,dy
real*8, dimension (:), allocatable :: aa,bb


nx = 256
ny = nx


ipr = 0

allocate(u(0:nx,0:ny))
allocate(v(0:nx,0:ny))
allocate(w(0:nx,0:ny))
allocate(rh(0:nx,0:ny))
allocate(p(0:nx,0:ny))
allocate(x(0:nx,0:ny))
allocate(y(0:nx,0:ny))


do ifile=nsnap_s,nsnap_e

jj = 0

do myid = 0,np-1
  
write(charID,'(i5)') myid       !index for each processor 
write(snapID,'(i5)') ifile      !index for time snapshot

! Read data file
fi2= "load_"// trim(adjustl(charID)) // '.dat'
open(unit=19, file=fi2)
read(19,*)myid2,nyl
close(19)


! Read data file

! Define the file name
filename = "vel_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.plt'

! Open the file and start reading the data
open(unit=19, file=filename)
read(19,*) 
read(19,*) 
read(19,*)
! Read velocity  data
do j = 0, nyl
do i = 0, nx
  	read(19,*) x(i,j+jj), y(i,j+jj), u(i,j+jj),v(i,j+jj)
end do
end do
close(19)

if (ipr.eq.1) then !defined pressure
! Define the file name
filename = "pressure_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.plt'

! Open the file and start reading the data
open(unit=19, file=filename)
read(19,*) 
read(19,*) 
read(19,*)
! Read density  data
do j = 0, nyl
do i = 0, nx
  	read(19,*) x(i,j+jj), y(i,j+jj), p(i,j+jj)
end do
end do
close(19)

else
  
do j = 0, nyl
do i = 0, nx
  	p(i,j+jj) = 0.0d0
end do
end do
  
end if

! Define the file name
filename = "density_"// trim(adjustl(snapID)) //'_' // trim(adjustl(charID)) // '.plt'

! Open the file and start reading the data
open(unit=19, file=filename)
read(19,*) 
read(19,*) 
read(19,*)
! Read density  data
do j = 0, nyl
do i = 0, nx
  	read(19,*) x(i,j+jj), y(i,j+jj), rh(i,j+jj)
end do
end do
close(19)


jj = nyl+jj

end do

! writing density field at cell centers:
! Define the file name
filename = "rho_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "rho"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', nx+1, ' j=', ny+1, ' f=point'
do j=0,ny
do i=0,nx
write(19,*)x(i,j),y(i,j),rh(i,j)
end do
end do
close(19)


!constant dx and dy
dx = x(2,1)-x(1,1)
dy = y(1,2)-y(1,1)


! compute vorticity
! v_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = v(i,j)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	w(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)

! u_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = u(i,j)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	w(i,j) = w(i,j) - bb(j)
	end do
end do
deallocate(aa,bb)

! writing vorticity field at cell centers:
! Define the file name
filename = "vorticity_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "Vorticity"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', nx+1, ' j=', ny+1, ' f=point'
do j=0,ny
do i=0,nx
write(19,*)x(i,j),y(i,j),w(i,j)
end do
end do
close(19)

!compute density weighted spectrum from velocity and density fields
call spectrum_den_2d(nx,ny,rh,u,v,ifile)

!compute spectrum from velocity field
call spectrum2d(nx,ny,u,v,ifile)

!compute structure function
call strfun2d(nx,ny,u,v,ifile)

!compute pdf distributions
!call pdfdist2d(nx,ny,rh,u,v,w,ifile)

!compute energy transfer from velocity field
!call energy_basic_2d(nx,ny,dx,dy,u,v,ifile)

!compute energy transfer for compressible flows
!call energy_ave_2d(nx,ny,dx,dy,u,v,ifile)

!compute energy transfer for compressible flows
!call energy_all_2d(nx,ny,dx,dy,u,v,rh,p,ifile)

!old post-processing routines:

!compute spectrum from vorticity field
!call spectrum2d_vorticity(nx,ny,w,ifile)
!appropriate one
!call spectrum2d_vorticity_new(nx,ny,w,ifile)
!compute spectrum from velocity field
!call spec(nx,ny,u,v,ifile)
!compute spectrum from vorticity field
!call specw(nx,ny,w,ifile)
!compute spectrum from vorticity data (the one we used before)
!call specnew(nx,ny,w,ifile)


end do

end

!-----------------------------------------------------------------!
!Compute energy transfer from velocity field 
!-----------------------------------------------------------------!
subroutine energy_basic_2d(nx,ny,dx,dy,u,v,ifile)
implicit none
integer::nx,ny,ifile
real*8 ::dx,dy
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
integer::i,j,p,np
real*8 ::kx(0:nx),ky(0:ny)
real*8,dimension(:),allocatable:: data1d,data2d,data3d,data4d,ae
real*8,dimension(:,:),allocatable::ee
integer,parameter::ndim=2
integer::nn(ndim),isign
real*8 ::temp,kr
character(80):: snapID,filename
real*8,dimension (:), allocatable :: aa,bb
real*8,dimension(:,:),allocatable :: ux,uy,vx,vy,ax,ay



!compute convective terms:
allocate(ux(0:nx,0:ny))
allocate(uy(0:nx,0:ny))

! u_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = u(i,j)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	ux(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)

! u_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = u(i,j)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	uy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ax(0:nx,0:ny))

do i=0,nx
do j=0,ny
ax(i,j) = u(i,j)*ux(i,j) + v(i,j)*uy(i,j)
end do
end do

deallocate(ux,uy)

allocate(vx(0:nx,0:ny))
allocate(vy(0:nx,0:ny))

! v_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = v(i,j)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	vx(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)


! v_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = v(i,j)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	vy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ay(0:nx,0:ny))

do i=0,nx
do j=0,ny
ay(i,j) = u(i,j)*vx(i,j) + v(i,j)*vy(i,j)
end do
end do

deallocate(vx,vy)


write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny))
allocate(data2d(2*nx*ny))
allocate(data3d(2*nx*ny))
allocate(data4d(2*nx*ny))

nn(1)= nx
nn(2)= ny

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


!finding fourier coefficients of u and v
!invese fourier transform
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j)
data1d(p+1)=0.0d0
data2d(p)  =v(i,j)
data2d(p+1)=0.0d0
data3d(p)  =ax(i,j)
data3d(p+1)=0.0d0
data4d(p)  =ay(i,j)
data4d(p+1)=0.0d0
p=p+2
end do
end do

deallocate(ax,ay)

isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)
call fourn(data3d,nn,ndim,isign)
call fourn(data4d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
data3d(p)=data3d(p)*temp
data4d(p)=data4d(p)*temp
end do

!TE(k)
allocate(ee(0:nx-1,0:ny-1))

p=1
do j=0,ny-1
do i=0,nx-1
ee(i,j)= data1d(p)*data3d(p) + data1d(p+1)*data3d(p+1) &
       + data2d(p)*data4d(p) + data2d(p+1)*data4d(p+1)
p=p+2
end do
end do

deallocate(data1d,data2d,data3d,data4d)

! energy flux
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(ae(np))

!summation!
do p=1,np
	ae(p)=0.0d0
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j))
        if (kr.gt.dfloat(p)) then
		ae(p)=ae(p)+ee(i,j)
		end if
	end do
    end do
end do

! Define the file name
filename = "eflux_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","P(k)"'
do p=1,np
write(802,*)dfloat(p),-ae(p)
end do
close(802)


deallocate(ee,ae)

return
end 

!-----------------------------------------------------------------!
!Compute energy transfer from velocity field
!-----------------------------------------------------------------!
subroutine energy_ave_2d(nx,ny,dx,dy,u,v,ifile)
implicit none
integer::nx,ny,ifile
real*8 ::dx,dy
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
integer::i,j,p,pp,np
real*8 ::kx(0:nx),ky(0:ny)
real*8,dimension(:),allocatable:: data1d,data2d,data3d,data4d,ae,as
real*8,dimension(:,:),allocatable::ee
integer,parameter::ndim=2
integer::nn(ndim),isign
real*8 ::temp,kr
character(80):: snapID,filename
real*8,dimension (:), allocatable :: aa,bb
real*8,dimension(:,:),allocatable :: ux,uy,vx,vy,ax,ay



!compute convective terms:
allocate(ux(0:nx,0:ny))
allocate(uy(0:nx,0:ny))

! u_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = u(i,j)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	ux(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)

! u_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = u(i,j)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	uy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ax(0:nx,0:ny))

do i=0,nx
do j=0,ny
ax(i,j) = u(i,j)*ux(i,j) + v(i,j)*uy(i,j)
end do
end do

deallocate(ux,uy)

allocate(vx(0:nx,0:ny))
allocate(vy(0:nx,0:ny))

! v_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = v(i,j)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	vx(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)


! v_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = v(i,j)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	vy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ay(0:nx,0:ny))

do i=0,nx
do j=0,ny
ay(i,j) = u(i,j)*vx(i,j) + v(i,j)*vy(i,j)
end do
end do

deallocate(vx,vy)


write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny))
allocate(data2d(2*nx*ny))
allocate(data3d(2*nx*ny))
allocate(data4d(2*nx*ny))

nn(1)= nx
nn(2)= ny

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


!finding fourier coefficients of u and v
!invese fourier transform
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j)
data1d(p+1)=0.0d0
data2d(p)  =v(i,j)
data2d(p+1)=0.0d0
data3d(p)  =ax(i,j)
data3d(p+1)=0.0d0
data4d(p)  =ay(i,j)
data4d(p+1)=0.0d0
p=p+2
end do
end do

deallocate(ax,ay)

isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)
call fourn(data3d,nn,ndim,isign)
call fourn(data4d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
data3d(p)=data3d(p)*temp
data4d(p)=data4d(p)*temp
end do

!TE(k)
allocate(ee(0:nx-1,0:ny-1))

p=1
do j=0,ny-1
do i=0,nx-1
ee(i,j)= data1d(p)*data3d(p) + data1d(p+1)*data3d(p+1) &
       + data2d(p)*data4d(p) + data2d(p+1)*data4d(p+1)
p=p+2
end do
end do

deallocate(data1d,data2d,data3d,data4d)

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(ae(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j)
		end if
	end do
    end do
end do

allocate(as(np))

!summation!

do p=1,np
as(p) = 0.0d0

	do pp=p,np
	as(p) = as(p) + ae(pp)
    end do
	
end do

! Define the file name
filename = "eflux_ave_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","P(k)"'
do p=1,np
write(802,*)dfloat(p),-as(p)
end do
close(802)


deallocate(ee,ae,as)

return
end 


!-----------------------------------------------------------------!
!Compute energy transfer from velocity, density, pressure fields 
!Han Liu, Z Xiao PRE 2016 
!-----------------------------------------------------------------!
subroutine energy_all_2d(nx,ny,dx,dy,u,v,rh,pr,ifile)
implicit none
integer::nx,ny,ifile
real*8 ::dx,dy
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
real*8 ::rh(0:nx,0:ny) 
real*8 ::pr(0:nx,0:ny) 
integer::i,j,p,pp,np
real*8 ::kx(0:nx),ky(0:ny)
real*8,dimension(:),allocatable:: data1d,data2d,data3d,data4d,ae,as,at
real*8,dimension(:,:),allocatable::ee
integer,parameter::ndim=2
integer::nn(ndim),isign
real*8 ::temp,kr
character(80):: snapID,filename
real*8,dimension (:), allocatable :: aa,bb
real*8,dimension(:,:),allocatable :: ux,uy,vx,vy,ax,ay

!--------------------------------!
!compute advective terms:
!--------------------------------!
allocate(ux(0:nx,0:ny))
allocate(uy(0:nx,0:ny))

! u_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = u(i,j)*dsqrt(dabs(rh(i,j)))
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	ux(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)

! u_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = u(i,j)*dsqrt(dabs(rh(i,j)))
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	uy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ax(0:nx,0:ny))

do i=0,nx
do j=0,ny
ax(i,j) = u(i,j)*ux(i,j) + v(i,j)*uy(i,j)
end do
end do

deallocate(ux,uy)

allocate(vx(0:nx,0:ny))
allocate(vy(0:nx,0:ny))

! v_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = v(i,j)*dsqrt(dabs(rh(i,j)))
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	vx(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)


! v_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = v(i,j)*dsqrt(dabs(rh(i,j)))
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	vy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ay(0:nx,0:ny))

do i=0,nx
do j=0,ny
ay(i,j) = u(i,j)*vx(i,j) + v(i,j)*vy(i,j)
end do
end do

deallocate(vx,vy)


write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny))
allocate(data2d(2*nx*ny))
allocate(data3d(2*nx*ny))
allocate(data4d(2*nx*ny))

nn(1)= nx
nn(2)= ny

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


!finding fourier coefficients of u and v
!invese fourier transform
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j)*dsqrt(dabs(rh(i,j)))
data1d(p+1)=0.0d0
data2d(p)  =v(i,j)*dsqrt(dabs(rh(i,j)))
data2d(p+1)=0.0d0
data3d(p)  =ax(i,j)
data3d(p+1)=0.0d0
data4d(p)  =ay(i,j)
data4d(p+1)=0.0d0
p=p+2
end do
end do

deallocate(ax,ay)

isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)
call fourn(data3d,nn,ndim,isign)
call fourn(data4d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
data3d(p)=data3d(p)*temp
data4d(p)=data4d(p)*temp
end do

!TE(k)
allocate(ee(0:nx-1,0:ny-1))

p=1
do j=0,ny-1
do i=0,nx-1
ee(i,j)= data1d(p)*data3d(p) + data1d(p+1)*data3d(p+1) &
       + data2d(p)*data4d(p) + data2d(p+1)*data4d(p+1)
p=p+2
end do
end do

deallocate(data1d,data2d,data3d,data4d)

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(ae(np))
allocate(at(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j)
		end if
	end do
    end do
end do

allocate(as(np))

!summation!

do p=1,np
as(p) = 0.0d0

	do pp=p,np
	as(p) = as(p) + ae(pp)
    end do

at(p) = as(p)	
end do

! Define the file name
filename = "eflux_ave_adv_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","P(k)"'
do p=1,np
write(802,*)dfloat(p),-as(p)
end do
close(802)

deallocate(ee,ae,as)



!-------------------------------------!
!compute dilatational terms:
!-------------------------------------!
allocate(ux(0:nx,0:ny))
allocate(vy(0:nx,0:ny))

! u_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = u(i,j)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	ux(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)

! v_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = v(i,j)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	vy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ax(0:nx,0:ny))

do i=0,nx
do j=0,ny
ax(i,j) = 0.5d0*u(i,j)*dsqrt(dabs(rh(i,j)))*(ux(i,j) + vy(i,j))
end do
end do

allocate(ay(0:nx,0:ny))

do i=0,nx
do j=0,ny
ay(i,j) = 0.5d0*v(i,j)*dsqrt(dabs(rh(i,j)))*(ux(i,j) + vy(i,j))
end do
end do

deallocate(ux,vy)


write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny))
allocate(data2d(2*nx*ny))
allocate(data3d(2*nx*ny))
allocate(data4d(2*nx*ny))


!finding fourier coefficients of u and v
!invese fourier transform
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j)*dsqrt(dabs(rh(i,j)))
data1d(p+1)=0.0d0
data2d(p)  =v(i,j)*dsqrt(dabs(rh(i,j)))
data2d(p+1)=0.0d0
data3d(p)  =ax(i,j)
data3d(p+1)=0.0d0
data4d(p)  =ay(i,j)
data4d(p+1)=0.0d0
p=p+2
end do
end do

deallocate(ax,ay)

isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)
call fourn(data3d,nn,ndim,isign)
call fourn(data4d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
data3d(p)=data3d(p)*temp
data4d(p)=data4d(p)*temp
end do

!TE(k)
allocate(ee(0:nx-1,0:ny-1))

p=1
do j=0,ny-1
do i=0,nx-1
ee(i,j)= data1d(p)*data3d(p) + data1d(p+1)*data3d(p+1) &
       + data2d(p)*data4d(p) + data2d(p+1)*data4d(p+1)
p=p+2
end do
end do

deallocate(data1d,data2d,data3d,data4d)

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(ae(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j)
		end if
	end do
    end do
end do

allocate(as(np))

!summation!

do p=1,np
as(p) = 0.0d0

	do pp=p,np
	as(p) = as(p) + ae(pp)
    end do

at(p) = at(p) + as(p)	
end do

! Define the file name
filename = "eflux_ave_dil_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","P(k)"'
do p=1,np
write(802,*)dfloat(p),-as(p)
end do
close(802)

deallocate(ee,ae,as)

!----------------------------------!
!compute pressure terms:
!----------------------------------!
allocate(ux(0:nx,0:ny))
allocate(uy(0:nx,0:ny))

! u_x
allocate(aa(0:nx),bb(0:nx))
do j=0,ny
	do i=0,nx
	aa(i) = pr(i,j)
	end do
		call c6dp(aa,bb,dx,nx)
	do i=0,nx
	ux(i,j) = bb(i)
	end do
end do
deallocate(aa,bb)

! u_y
allocate(aa(0:ny),bb(0:ny))
do i=0,nx
	do j=0,ny
	aa(j) = pr(i,j)
	end do
		call c6dp(aa,bb,dy,ny)
	do j=0,ny
	uy(i,j) = bb(j)
	end do
end do
deallocate(aa,bb)

allocate(ax(0:nx,0:ny))

do i=0,nx
do j=0,ny
ax(i,j) = 0.5d0*ux(i,j)/dsqrt(dabs(rh(i,j)))
end do
end do

allocate(ay(0:nx,0:ny))

do i=0,nx
do j=0,ny
ay(i,j) = 0.5d0*uy(i,j)/dsqrt(dabs(rh(i,j)))
end do
end do

deallocate(ux,uy)


write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny))
allocate(data2d(2*nx*ny))
allocate(data3d(2*nx*ny))
allocate(data4d(2*nx*ny))


!finding fourier coefficients of u and v
!invese fourier transform
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j)*dsqrt(dabs(rh(i,j)))
data1d(p+1)=0.0d0
data2d(p)  =v(i,j)*dsqrt(dabs(rh(i,j)))
data2d(p+1)=0.0d0
data3d(p)  =ax(i,j)
data3d(p+1)=0.0d0
data4d(p)  =ay(i,j)
data4d(p+1)=0.0d0
p=p+2
end do
end do

deallocate(ax,ay)

isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)
call fourn(data3d,nn,ndim,isign)
call fourn(data4d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
data3d(p)=data3d(p)*temp
data4d(p)=data4d(p)*temp
end do

!TE(k)
allocate(ee(0:nx-1,0:ny-1))

p=1
do j=0,ny-1
do i=0,nx-1
ee(i,j)= data1d(p)*data3d(p) + data1d(p+1)*data3d(p+1) &
       + data2d(p)*data4d(p) + data2d(p+1)*data4d(p+1)
p=p+2
end do
end do

deallocate(data1d,data2d,data3d,data4d)

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(ae(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j)
		end if
	end do
    end do
end do

allocate(as(np))

!summation!

do p=1,np
as(p) = 0.0d0

	do pp=p,np
	as(p) = as(p) + ae(pp)
    end do

at(p) = at(p) + as(p)	
end do

! Define the file name
filename = "eflux_ave_pre_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","P(k)"'
do p=1,np
write(802,*)dfloat(p),-as(p)
end do
close(802)

! Define the file name
filename = "eflux_ave_tot_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","P(k)"'
do p=1,np
write(802,*)dfloat(p),-at(p)
end do
close(802)

deallocate(ee,ae,as,at)


return
end 



!-----------------------------------------------------------------!
!Compute density weighted spectrum from 2D velocity and density fields 
!-----------------------------------------------------------------!
subroutine spectrum_den_2d(nx,ny,rh,u,v,ifile)
implicit none
integer::nx,ny,ifile
real*8 ::rh(0:nx,0:ny) 
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
integer::i,j,p,np
real*8 ::kx(0:nx),ky(0:ny)
real*8,dimension(:),allocatable:: data1d,data2d,ae
real*8,dimension(:,:),allocatable::ee
integer,parameter::ndim=2
integer::nn(ndim),isign
real*8 ::temp,kr
character(80):: snapID,filename


write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny))
allocate(data2d(2*nx*ny))

nn(1)= nx
nn(2)= ny

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


!finding fourier coefficients of u and v
!invese fourier transform
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =dsqrt(dabs(rh(i,j)))*u(i,j)
data1d(p+1)=0.0d0
data2d(p)  =dsqrt(dabs(rh(i,j)))*v(i,j)
data2d(p+1)=0.0d0
p=p+2
end do
end do


isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
end do

!Density weighted Energy density
allocate(ee(0:nx-1,0:ny-1))

p=1
do j=0,ny-1
do i=0,nx-1
ee(i,j)=0.5d0*(data1d(p)*data1d(p) + data1d(p+1)*data1d(p+1) &
              +data2d(p)*data2d(p) + data2d(p+1)*data2d(p+1))
p=p+2
end do
end do

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(ae(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j)
		end if
	end do
    end do
end do

! Define the file name
filename = "spectrum_den_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","E(k)"'
do p=1,np
write(802,*)dfloat(p),ae(p)
end do
close(802)

open(15,file="spectrum_k3.plt")
write(15,*) 'variables ="k","E(k)"'
write(15,*)2.0d0, 1.0d0
write(15,*)200.0d0, 1.0d-6
close(15)

deallocate(data1d,data2d,ee,ae)

return
end 




!-----------------------------------------------------------------!
!Compute spectrum from 2D velocity field by definition of Kida
!S Kida, Y. Murakami, et al. 1990
!-----------------------------------------------------------------!
subroutine spectrum2d(nx,ny,u,v,ifile)
implicit none
integer::nx,ny,ifile
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
integer::i,j,p,np
real*8 ::kx(0:nx),ky(0:ny)
real*8,dimension(:),allocatable:: data1d,data2d,ae
real*8,dimension(:,:),allocatable::ee
integer,parameter::ndim=2
integer::nn(ndim),isign
real*8 ::temp,kr
character(80):: snapID,filename


write(snapID,'(i5)') ifile      !index for time snapshot

allocate(data1d(2*nx*ny))
allocate(data2d(2*nx*ny))

nn(1)= nx
nn(2)= ny

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


!finding fourier coefficients of u and v
!invese fourier transform
p=1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j)
data1d(p+1)=0.0d0
data2d(p)  =v(i,j)
data2d(p+1)=0.0d0
p=p+2
end do
end do


isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
data2d(p)=data2d(p)*temp
end do

!Energy density
allocate(ee(0:nx-1,0:ny-1))

p=1
do j=0,ny-1
do i=0,nx-1
ee(i,j)=0.5d0*(data1d(p)*data1d(p) + data1d(p+1)*data1d(p+1) &
              +data2d(p)*data2d(p) + data2d(p+1)*data2d(p+1))
p=p+2
end do
end do

! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(ae(np))

!averaging!
do p=1,np
	ae(p)=0.0d0
	do j=0,ny-1
	do i=0,nx-1
    	kr=dsqrt(kx(i)*kx(i)+ky(j)*ky(j))
        if (kr.ge.(dfloat(p)-0.5d0).and.kr.lt.(dfloat(p)+0.5d0)) then
		ae(p)=ae(p)+ee(i,j)
		end if
	end do
    end do
end do

! Define the file name
filename = "spectrum_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="k","E(k)"'
do p=1,np
write(802,*)dfloat(p),ae(p)
end do
close(802)

open(15,file="spectrum_k3.plt")
write(15,*) 'variables ="k","E(k)"'
write(15,*)2.0d0, 1.0d0
write(15,*)200.0d0, 1.0d-6
close(15)

deallocate(data1d,data2d,ee,ae)

return
end 

!-----------------------------------------------------------------!
!Compute structure functions from 2D velocity field by definition 
!Boffetta, Musacchio, PRE 2010
!Kramer et al., PRE 2011
!-----------------------------------------------------------------!
subroutine strfun2d(nx,ny,u,v,ifile)
implicit none
integer::nx,ny,ifile
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
integer::i,j,ii,jj,ir,jr
real*8 ::dlu2(1:nx/2),dtv2(1:nx/2),dlu3(1:nx/2),dtv3(1:nx/2)
real*8 ::dlv2(1:ny/2),dtu2(1:ny/2),dlv3(1:ny/2),dtu3(1:ny/2)
real*8 ::dl2(1:ny/2),dt2(1:ny/2),dl3(1:ny/2),dt3(1:ny/2)
character(80):: snapID,filename



write(snapID,'(i5)') ifile      !index for time snapshot


do ii=1,nx/2
  	
	dlu2(ii) = 0.0d0
    dtv2(ii) = 0.0d0
    
    dlu3(ii) = 0.0d0
    dtv3(ii) = 0.0d0
    
	do j=1,ny
	do i=1,nx
    
    	if (i+ii.ge.nx) then
        ir = ii-nx
        else
        ir = ii
        end if
        
		dlu2(ii)=dlu2(ii)+(dabs(u(i+ir,j)-u(i,j)))**2 !longitudinal
        dtv2(ii)=dtv2(ii)+(dabs(v(i+ir,j)-v(i,j)))**2 !transverse
            
		dlu3(ii)=dlu3(ii)+(dabs(u(i+ir,j)-u(i,j)))**3 !longitudinal
        dtv3(ii)=dtv3(ii)+(dabs(v(i+ir,j)-v(i,j)))**3 !transverse
        
	end do
	end do
    	dlu2(ii) = dlu2(ii)/dfloat(nx*ny)
        dtv2(ii) = dtv2(ii)/dfloat(nx*ny)

        dlu3(ii) = dlu3(ii)/dfloat(nx*ny)
        dtv3(ii) = dtv3(ii)/dfloat(nx*ny)
end do

do jj=1,ny/2

  	dlv2(jj) = 0.0d0
  	dtu2(jj) = 0.0d0

    dlv3(jj) = 0.0d0
  	dtu3(jj) = 0.0d0
    
	do j=1,ny
	do i=1,nx
    
    	if (j+jj.ge.ny) then
        jr = jj-ny
        else
        jr = jj
        end if
        
		dlv2(jj)=dlv2(jj)+(dabs(v(i,j+jr)-v(i,j)))**2 !longitudinal
        dtu2(jj)=dtu2(jj)+(dabs(u(i,j+jr)-u(i,j)))**2 !transverse
 
		dlv3(jj)=dlv3(jj)+(dabs(v(i,j+jr)-v(i,j)))**3 !longitudinal
        dtu3(jj)=dtu3(jj)+(dabs(u(i,j+jr)-u(i,j)))**3 !transverse
                   
	end do
	end do

        dlv2(jj) = dlv2(jj)/dfloat(nx*ny)
        dtu2(jj) = dtu2(jj)/dfloat(nx*ny)

        dlv3(jj) = dlv3(jj)/dfloat(nx*ny)
        dtu3(jj) = dtu3(jj)/dfloat(nx*ny)

        dl2(jj) = 0.5d0*(dlv2(jj) + dlu2(jj))
        dt2(jj) = 0.5d0*(dtv2(jj) + dtu2(jj))

        dl3(jj) = 0.5d0*(dlv3(jj) + dlu3(jj))
        dt3(jj) = 0.5d0*(dtv3(jj) + dtu3(jj))
end do



  

! Define the file name
filename = "str_long_2nd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dfloat(jj)/dfloat(ny),dl2(jj)
end do
close(802)

! Define the file name
filename = "str_tran_2nd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dfloat(jj)/dfloat(ny),dt2(jj)
end do
close(802)

! Define the file name
filename = "str_long_3rd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dfloat(jj)/dfloat(ny),dl3(jj)
end do
close(802)

! Define the file name
filename = "str_tran_3rd_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=802, file=filename)
write(802,*)'variables ="r","F(r)"'
do jj=1,ny/2
write(802,*)dfloat(jj)/dfloat(ny),dt3(jj)
end do
close(802)



return
end 



!-----------------------------------------------------------------!
!Spectrum from vorticity data
!-----------------------------------------------------------------!
subroutine spectrum2d_vorticity(nx,ny,w,ifile)
implicit none
integer ::nx,ny,ifile
double precision::w(0:nx,0:ny)
double precision::pi,temp
integer::i,j,p,np,ic
double precision::kx(0:nx),ky(0:ny),kr
double precision,parameter:: tiny=1.0d-10
double precision,dimension(:),allocatable:: data1d,en
double precision,dimension(:,:),allocatable::es
integer,parameter::ndim=2
integer::nn(ndim),isign
character(80):: snapID,filename

write(snapID,'(i5)') ifile      !index for time snapshot


allocate(data1d(2*nx*ny))

pi = 4.0d0*datan(1.0d0)

nn(1)= nx
nn(2)= ny

!finding fourier coefficients of w 
!invese fourier transform
!find the vorticity in Fourier space
p=1
do j=0,ny-1  
do i=0,nx-1   
	data1d(p)   =  w(i,j)
	data1d(p+1) =  0.0d0    
p = p + 2
end do
end do

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
end do

!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)


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
kx(0) = tiny

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
ky(0) = tiny


!Energy spectrum (for all wavenumbers), amplitude
allocate(es(0:nx-1,0:ny-1))
p=1
do j=0,ny-1
do i=0,nx-1
kr = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
es(i,j) = pi*(data1d(p)*data1d(p) + data1d(p+1)*data1d(p+1))/kr
p = p + 2
end do
end do


! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(en(np))

do p=1,np
en(p) = 0.0d0
ic = 0
do j=0,ny-1
do i=0,nx-1
	kr = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
    if(kr.ge.(dfloat(p)-0.5d0).and.kr.le.(dfloat(p)+0.5d0)) then
    ic = ic + 1
    en(p) = en(p) + es(i,j)
    end if
end do
end do
en(p) = en(p) / dfloat(ic)
end do

! Define the file name
filename = "spectrum_vorticity_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=902, file=filename)
write(902,*)'variables ="k","E(k)"'
do p=1,np
write(902,*)dfloat(p),en(p)
end do
close(902)

deallocate(data1d,es,en)

return
end 


!-----------------------------------------------------------------!
!Spectrum from vorticity data
!-----------------------------------------------------------------!
subroutine spectrum2d_vorticity_new(nx,ny,w,ifile)
implicit none
integer ::nx,ny,ifile
double precision::w(0:nx,0:ny)
double precision::temp
integer::i,j,p,np,ic
double precision::kx(0:nx),ky(0:ny),kr
double precision,parameter:: tiny=1.0d-10
double precision,dimension(:),allocatable:: data1d,en
double precision,dimension(:,:),allocatable::es
integer,parameter::ndim=2
integer::nn(ndim),isign
character(80):: snapID,filename

write(snapID,'(i5)') ifile      !index for time snapshot


allocate(data1d(2*nx*ny))



nn(1)= nx
nn(2)= ny

!finding fourier coefficients of w 
!invese fourier transform
!find the vorticity in Fourier space
p=1
do j=0,ny-1  
do i=0,nx-1   
	data1d(p)   =  w(i,j)
	data1d(p+1) =  0.0d0    
p = p + 2
end do
end do

temp = 1.0d0/dfloat(nx*ny)

!normalize
do p=1,2*nx*ny
data1d(p)=data1d(p)*temp
end do

!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)


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
kx(0) = tiny

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
ky(0) = tiny


!Energy spectrum (for all wavenumbers), amplitude
allocate(es(0:nx-1,0:ny-1))
p=1
do j=0,ny-1
do i=0,nx-1
kr = kx(i)*kx(i) + ky(j)*ky(j)
es(i,j) = 0.5d0*(data1d(p)*data1d(p) + data1d(p+1)*data1d(p+1))/kr
p = p + 2
end do
end do


! band-avaraged energy spectrum
np = nint(dsqrt((kx(nx/2))**2 + (ky(ny/2))**2)) 

allocate(en(np))

do p=1,np
en(p) = 0.0d0
ic = 0
do j=0,ny-1
do i=0,nx-1
	kr = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
    if(kr.ge.(dfloat(p)-0.5d0).and.kr.le.(dfloat(p)+0.5d0)) then
    ic = ic + 1
    en(p) = en(p) + es(i,j)
    end if
end do
end do
en(p) = en(p)
end do

! Define the file name
filename = "spectrum_vorticity_new_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=902, file=filename)
write(902,*)'variables ="k","E(k)"'
do p=1,np
write(902,*)dfloat(p),en(p)
end do
close(902)

deallocate(data1d,es,en)

return
end 

!-----------------------------------------------------------------!
!Compute spectrum by definition of Kida
!S Kida, Y. Murakami, et al. 1990
!-----------------------------------------------------------------!
subroutine spec_kida(nx,ny,u,v,ifile)
implicit none
integer::nx,ny,n,ifile
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
integer::i,j,k,k1,k2
real*8,parameter:: tiny=1.0d-10,tiny2=1.0d-14
real*8,dimension(:),allocatable:: data1d,data2d,u1,u2,kk,ee,ae
integer,parameter::ndim=2
integer::nn(ndim),isign
real*8 ::temp
character(80):: snapID,filename



write(snapID,'(i5)') ifile      !index for time snapshot



allocate(data1d(2*nx*ny),data2d(2*nx*ny),u1(nx*ny),u2(nx*ny),kk(nx*ny),ee(nx*ny))


nn(1)= nx
nn(2)= ny

!finding fourier coefficients of u and v
!invese fourier transform
k=1
do j=1,ny
do i=1,nx
data1d(k)  =u(i,j)
data1d(k+1)=0.0d0
data2d(k)  =v(i,j)
data2d(k+1)=0.0d0
k=k+2
end do
end do


isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)

temp = 1.0d0/dfloat(nx*ny)

do j=1,2*nx*ny
data1d(j)=data1d(j)*temp
data2d(j)=data2d(j)*temp
end do

! absolute values for fourier coefficients, squared
k=1
do j=1,nx*ny
u1(j)=data1d(k)*data1d(k) + data1d(k+1)*data1d(k+1)
u2(j)=data2d(k)*data2d(k) + data2d(k+1)*data2d(k+1)
ee(j)=0.5d0*(u1(j)+u2(j))
k=k+2
end do


! wave numbers, real number absolute
k=1
do k2 = 0,ny/2
	do k1= 0,nx/2 
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
	do k1= -nx/2+1,-1
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
end do

do k2 =-ny/2+1,-1
	do k1= 0,nx/2 
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
	do k1=-nx/2+1,-1
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
end do


! avaraged energy spectrum
n=int(sqrt(float(nx*ny))*sqrt(2.0)/2.0 )-1
allocate(ae(n))


!averaging!
do i=1,n
	ae(i)=0.0d0
	do j=1,nx*ny
        if (kk(j).ge.(dfloat(i)-0.5d0).and.kk(j).lt.(dfloat(i)+0.5d0)) then
		ae(i)=ae(i)+ee(j)
		end if
	end do
end do

! Define the file name
filename = "spectrum_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=800, file=filename)
write(800,*)'variables ="k","E(k)"'
do i=1,n
write(800,*)dfloat(i),ae(i)
end do
close(800)

open(15,file="spectrum_k3.plt")
write(15,*) 'variables ="k","E(k)"'
write(15,*)2.0d0, 1.0d0
write(15,*)200.0d0, 1.0d-6
close(15)

deallocate(data1d,data2d,u1,u2,kk,ee,ae)




return
end 


!-----------------------------------------------------------------!
!Compute spectrum
!-----------------------------------------------------------------!
subroutine spec(nx,ny,u,v,ifile)
implicit none
integer::nx,ny,n,ic,ifile
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny) 
integer::i,j,k,k1,k2

real*8,parameter:: tiny=1.0d-10,tiny2=1.0d-14
real*8,dimension(:),allocatable:: data1d,data2d,u1,u2,kk,ee,ae

integer,parameter::ndim=2
integer::nn(ndim),isign
real*8 ::pi
character(80):: snapID,filename

pi = 4.0d0*datan(1.0d0)

write(snapID,'(i5)') ifile      !index for time snapshot



allocate(data1d(2*nx*ny),data2d(2*nx*ny),u1(nx*ny),u2(nx*ny),kk(nx*ny),ee(nx*ny))


nn(1)= nx
nn(2)= ny

!finding fourier coefficients of u and v
!invese fourier transform
k=1
do j=1,ny
do i=1,nx
data1d(k)  =u(i,j)
data1d(k+1)=0.0d0
data2d(k)  =v(i,j)
data2d(k+1)=0.0d0
k=k+2
end do
end do


isign=-1
call fourn(data1d,nn,ndim,isign)
call fourn(data2d,nn,ndim,isign)

do j=1,2*nx*ny
data1d(j)=data1d(j) /dfloat(nx*ny)
data2d(j)=data2d(j) /dfloat(nx*ny)
end do

! absolute values for fourier coefficients
k=1
do j=1,nx*ny
u1(j)=data1d(k)*data1d(k) + data1d(k+1)*data1d(k+1)
u2(j)=data2d(k)*data2d(k) + data2d(k+1)*data2d(k+1)
k=k+2
end do


! wave numbers
k=1
do k2 = 0,ny/2
	do k1= 0,nx/2 
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
	do k1= -nx/2+1,-1
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
end do

do k2 =-ny/2+1,-1
	do k1= 0,nx/2 
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
	do k1=-nx/2+1,-1
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
end do

! energy spectrum 

do j=1,nx*ny	
	ee(j)=0.5d0*(u1(j)+u2(j))  !u1 and u2 are squared data
end do

! Define the file name
filename = "spectrum_scattered_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=700, file=filename)
write(700,*)'variables ="k","E"'
do j=1,nx*ny
write(700,*)kk(j),ee(j)
end do
close(700)


! avaraged energy spectrum
n=int(sqrt(float(nx*ny))*sqrt(2.0)/2.0 )-1
allocate(ae(n))


!averaging!
do i=1,n
	ae(i)=0.0d0
	ic=0
	do j=1,nx*ny
		if (kk(j).ge.dfloat(i-1).and.kk(j).le.(dfloat(i)-tiny)) then
		ic=ic+1
		ae(i)=ae(i)+ee(j)
		end if
	end do
	ae(i)=ae(i)/dfloat(ic)
end do

! Define the file name
filename = "spectrum_averaged_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=800, file=filename)
write(800,*)'variables ="k","E"'
do i=1,n
write(800,*)dfloat(i),ae(i)*(2.0d0*pi*(dfloat(i)))	
end do
close(800)

! Define the file name
filename = "spectrum_averaged_new"// trim(adjustl(snapID)) //'.plt'
  
open(unit=805, file=filename)
write(805,*)'variables ="k","E"'
do i=1,n
write(805,*)dfloat(i),ae(i)
end do
close(805)


open(15,file="k3.plt")
write(15,*) 'variables ="k","E"'
write(15,*)2.0d0, 1.0d0
write(15,*)200.0d0, 1.0d-6
close(15)

deallocate(data1d,data2d,u1,u2,kk,ee,ae)




return
end 

!-----------------------------------------------------------------!
!Compute spectrum from vorticity
!-----------------------------------------------------------------!
subroutine specw(nx,ny,w,ifile)
implicit none
integer ::nx,ny,n,ic,ifile
double precision::pi
integer::i,j,k,k1,k2
real*8 ::w(0:nx,0:ny) 

real*8,parameter:: tiny=1.0d-10,tiny2=1.0d-14
real*8,dimension(:),allocatable:: data1d,wa2,kk,ee,ae

integer,parameter::ndim=2
integer::nn(ndim),isign
character(80):: snapID,filename

write(snapID,'(i5)') ifile      !index for time snapshot

pi = 4.0d0*datan(1.0d0)

allocate(data1d(2*nx*ny),wa2(nx*ny),kk(nx*ny),ee(nx*ny))


nn(1)= nx
nn(2)= ny

!finding fourier coefficients of w 
!invese fourier transform
k=1
do j=1,ny
do i=1,nx
data1d(k)  =w(i,j)
data1d(k+1)=0.0d0
k=k+2
end do
end do


isign=-1
call fourn(data1d,nn,ndim,isign)

do j=1,2*nx*ny
data1d(j)=data1d(j)/dfloat(nx*ny)
end do

! absolute value of vorticity coefficients in fourier space
k=1
do j=1,nx*ny
wa2(j)=data1d(k)*data1d(k) + data1d(k+1)*data1d(k+1)
k=k+2
end do


! wave numbers
k=1
do k2 = 0,ny/2
	do k1= 0,nx/2 
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
	do k1= -nx/2+1,-1
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
end do

do k2 =-ny/2+1,-1
	do k1= 0,nx/2 
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
	do k1=-nx/2+1,-1
	kk(k)  = dsqrt(dfloat(k1)*dfloat(k1) + dfloat(k2)*dfloat(k2))
	k=k+1	
	end do
end do

! energy spectrum logaritmic scale
filename = "spectrum_vor_scattered_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=701, file=filename)
write(701,*)'variables ="log(k)","log(e)"'

k=0
do j=1,nx*ny
if(dabs(kk(j)).ge.tiny) then
	ee(j)=wa2(j)*pi/kk(j)
	if (ee(j).le.tiny2) ee(j)=tiny2
	k=k+1
	kk(k) = dlog10(kk(j))
	ee(k) = dlog10(ee(j))
	write(701,103)kk(k),ee(k)
end if
end do
close(701)



! avaraged energy spectrum

n=int(sqrt(float(nx*ny))*sqrt(2.0)/2.0 )-1
allocate(ae(n))


!averaging!
do i=1,n
	ae(i)=0.0d0
	ic=0
	do j=1,nx*ny
		if (kk(j).ge.dlog10(dfloat(i)-0.5d0).and.kk(j).le.dlog10(dfloat(i)+0.5d0)) then
		ic=ic+1
		ae(i)=ae(i)+ee(j)
		end if
	end do
	ae(i)=ae(i)/dfloat(ic)
end do

filename = "spectrum_vor_averaged_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=702, file=filename)
write(702,*)'variables ="log(k)","log(e)"'
do i=1,n
write(702,*)dlog10(dfloat(i)),ae(i)
end do

close(702)

deallocate(data1d,wa2,kk,ee,ae)




103 format(2e16.6)

return
end 


!-----------------------------------------------------------------!
!Spectrum new
!-----------------------------------------------------------------!
subroutine specnew(nx,ny,w,ifile)
implicit none
integer ::nx,ny,ifile
double precision::w(0:nx,0:ny)
double precision::pi
integer::i,j,k,n,ic
double precision::kx(0:nx-1),ky(0:ny-1),kk
double precision,parameter:: tiny=1.0d-10
double precision,dimension(:),allocatable:: data1d,en
double precision,dimension(:,:),allocatable::es
integer,parameter::ndim=2
integer::nn(ndim),isign
character(80):: snapID,filename

write(snapID,'(i5)') ifile      !index for time snapshot


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

! energy spectrum logaritmic scale
filename = "spec_new_vor_scattered_"// trim(adjustl(snapID)) //'.plt'
open(unit=901, file=filename)
write(901,*)'variables ="k","E"'
do j=1,ny-1
do i=1,nx-1
kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
write(901,103) kk,es(i,j)
end do
end do
close(901)


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

! Define the file name
filename = "spec_new_vor_averaged_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=902, file=filename)
write(902,*)'variables ="k","E"'
do k=1,n
write(902,103)dfloat(k),en(k)
end do
close(902)


deallocate(data1d,es,en)

103 format(2e16.6)

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


!-----------------------------------------------------------------!
!Probability density function writing
!very naive way
!-----------------------------------------------------------------!
subroutine pdfdist2d(nx,ny,rh,u,v,w,ifile)
implicit none
integer::nx,ny,ifile,l,ii
real*8 ::rh(0:nx,0:ny)
real*8 ::u(0:nx,0:ny) 
real*8 ::v(0:nx,0:ny)
real*8 ::w(0:nx,0:ny)  
real*8,allocatable ::du(:),uu(:),pu(:),nu(:) 
character(80):: snapID,filename
integer::npdf,ndata,i,j


write(snapID,'(i5)') ifile      !index for time snapshot

npdf = 512  !number of bin in pdf computation


! writing pdf for vorticity 
ndata = nx*ny
allocate(du(ndata),uu(npdf),pu(npdf),nu(npdf))
ii=1
do j = 1,ny
do i = 1,nx
du(ii) = w(i,j)
ii = ii + 1
end do
end do

call pdf2d(ndata,npdf,du,uu,pu,nu)
! Define the file name
filename = "pdf_vorticity_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "w", "PDF","G"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', npdf, ' f=point'
do l=1,npdf
write(19,*)uu(l),pu(l),nu(l)
end do
close(19)
deallocate(du,uu,pu,nu)

! writing pdf for vorticity difference
ndata = 2*nx*ny
allocate(du(ndata),uu(npdf),pu(npdf),nu(npdf))
ii=1
do j = 1,ny
do i = 1,nx
du(ii) = w(i,j)-w(i-1,j)
du(ii+1)=w(i,j)-w(i,j-1)
ii = ii + 2
end do
end do

call pdf2d(ndata,npdf,du,uu,pu,nu)
! Define the file name
filename = "pdf_vorticity_dif_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "dw", "PDF","G"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', npdf, ' f=point'
do l=1,npdf
write(19,*)uu(l),pu(l),nu(l)
end do
close(19)
deallocate(du,uu,pu,nu)



! writing pdf for density 
ndata = nx*ny
allocate(du(ndata),uu(npdf),pu(npdf),nu(npdf))
ii=1
do j = 1,ny
do i = 1,nx
du(ii) = rh(i,j) 
ii=ii+1
end do
end do

call pdf2d(ndata,npdf,du,uu,pu,nu)
! Define the file name
filename = "pdf_rho_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "rho", "PDF","G"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', npdf, ' f=point'
do l=1,npdf
write(19,*)uu(l),pu(l),nu(l)
end do
close(19)
deallocate(du,uu,pu,nu)


! writing pdf for density difference in x 
ndata = 2*nx*ny
allocate(du(ndata),uu(npdf),pu(npdf),nu(npdf))
ii=1
do j = 1,ny
do i = 1,nx
du(ii) = rh(i,j)-rh(i-1,j)  
du(ii+1) = rh(i,j)-rh(i,j-1) 
ii=ii+2
end do
end do

call pdf2d(ndata,npdf,du,uu,pu,nu)
! Define the file name
filename = "pdf_rho_dif_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "dr", "PDF","G"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', npdf, ' f=point'
do l=1,npdf
write(19,*)uu(l),pu(l),nu(l)
end do
close(19)
deallocate(du,uu,pu,nu)


! writing pdf for velocity difference in longitudinal 
ndata = 2*nx*ny
allocate(du(ndata),uu(npdf),pu(npdf),nu(npdf))
ii=1
do j = 1,ny
do i = 1,nx
du(ii) = u(i,j)-u(i-1,j) 
du(ii+1) = v(i,j)-v(i,j-1)  
ii=ii+2 
end do
end do

call pdf2d(ndata,npdf,du,uu,pu,nu)
! Define the file name
filename = "pdf_vel_longitudinal_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "du_longitudinal", "PDF","G"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', npdf, ' f=point'
do l=1,npdf
write(19,*)uu(l),pu(l),nu(l)
end do
close(19)
deallocate(du,uu,pu,nu)

! writing pdf for velocity difference in transverse 
ndata=2*nx*ny
allocate(du(ndata),uu(npdf),pu(npdf),nu(npdf))
ii=1
do j = 1,ny
do i = 1,nx
du(ii) = u(i,j)-u(i,j-1) 
du(ii+1) = v(i,j)-v(i-1,j) 
ii=ii+2
end do
end do

call pdf2d(ndata,npdf,du,uu,pu,nu)
! Define the file name
filename = "pdf_vel_transverse_"// trim(adjustl(snapID)) //'.plt'
  
open(unit=19, file=filename)
write(19,*) 'variables = "du_transverse", "PDF","G"'
write(19,*) 'zone T=','Zone_'// trim(adjustl(snapID)), &
            ' i=', npdf, ' f=point'
do l=1,npdf
write(19,*)uu(l),pu(l),nu(l)
end do
close(19)
deallocate(du,uu,pu,nu)



return
end

!-----------------------------------------------------------------!
!Probability density function computing
!-----------------------------------------------------------------!
subroutine pdf2d(ndata,npdf,du,uu,pu,nu)
implicit none
integer::ndata,npdf
real*8 ::du(ndata),uu(npdf),pu(npdf),nu(npdf),uf(0:npdf)  
integer::ii,l
real*8 ::dmax,dmin,sum,mean,sd,pi

pi = 4.0d0*datan(1.0d0)

!dmax = maxval(du)
!dmin = minval(du)

!normalize the data (z-score):

!compute mean:
mean = 0.0d0
do ii=1,ndata
mean = mean + du(ii)
end do
mean = mean/dfloat(ndata)

!standard deviation:
sd = 0.0d0
do ii=1,ndata
sd = sd + (du(ii)-mean)**2
end do
sd = sd/dfloat(ndata)
sd = dsqrt(sd)

dmax = min(mean + sd*4.0d0, maxval(du))
dmin = max(mean - sd*4.0d0, minval(du))

do l = 1,npdf
  pu(l) = 0.0d0
end do

do l = 0,npdf
  !if (l.le.npdf/2) then
  !uf(l) = dmin + ((dmax-dmin)/2.0d0)*dsin(dfloat(l)/dfloat(npdf)*pi)
  !else
  !uf(l) = dmax - ((dmax-dmin)/2.0d0)*dsin(dfloat(l)/dfloat(npdf)*pi)
  !end if
  uf(l) = dmin + (dmax-dmin)*dfloat(l)/dfloat(npdf)
end do


do l = 1,npdf
do ii=1,ndata
if (du(ii).lt.uf(l).and.du(ii).ge.uf(l-1)) then
pu(l) = pu(l)+1.0d0
end if
end do
uu(l) = 0.5d0*(uf(l)+uf(l-1))
nu(l) = 1.0d0/dsqrt(2.0d0*pi*sd**2)*dexp(-((uu(l)-mean)**2)/(2.0d0*(sd**2)))
end do

sum = 0.0d0
do l = 1,npdf
  sum = sum + pu(l)*(uf(l)-uf(l-1))
end do

if (sum.le.1.0d0) sum = 1.0d0

!normalize to give sum=1.0d0
do l = 1,npdf
  pu(l) = pu(l)/sum
end do

return
end