!---------------------------------------------------------------------------!
!Computing true SGS from velocity field
!---------------------------------------------------------------------------!
!Omer San, Oklahoma State University, cfdlab.osu@gmail.com 
!Updated: Jan 30, 2018
!---------------------------------------------------------------------------!

program sgscompute3d
implicit none
integer::nx,ny,nz,nxc,nyc,nzc,nr,i,j,k,nbin,if1,ns,nf
double precision,dimension(:,:,:),allocatable::x,y,z
double precision,dimension(:,:,:),allocatable::u,v,w
double precision,dimension(:,:,:),allocatable::xc,yc,zc
double precision,dimension(:,:,:),allocatable::uc,vc,wc
double precision,dimension(:,:,:),allocatable::t11,t12,t13,t22,t23,t33
character(80):: snapID,corID,filename

!number of files:
ns=1
nf=5

!DNS resolution
nx=512
ny=nx
nz=nx

!LES grid ratio:
nr = 8

write(corID,'(i5)') nr  !index for coarsining ratio

!LES resolution
nxc=nx/nr
nyc=nxc
nzc=nxc

!perform for all data sets:
do if1=ns,nf
  
!DNS grid
allocate(x(0:nx,0:ny,0:nz))
allocate(y(0:nx,0:ny,0:nz))
allocate(z(0:nx,0:ny,0:nz))

!DNS velocity components
allocate(u(0:nx,0:ny,0:nz))
allocate(v(0:nx,0:ny,0:nz))
allocate(w(0:nx,0:ny,0:nz))

!LES grid
allocate(xc(0:nxc,0:nyc,0:nzc))
allocate(yc(0:nxc,0:nyc,0:nzc))
allocate(zc(0:nxc,0:nyc,0:nzc))

!LES velocity components
allocate(uc(0:nxc,0:nyc,0:nzc))
allocate(vc(0:nxc,0:nyc,0:nzc))
allocate(wc(0:nxc,0:nyc,0:nzc))

!true SGS stress (deviatoric part) on LES grid
allocate(t11(0:nxc,0:nyc,0:nzc))
allocate(t12(0:nxc,0:nyc,0:nzc))
allocate(t13(0:nxc,0:nyc,0:nzc))
allocate(t22(0:nxc,0:nyc,0:nzc))
allocate(t23(0:nxc,0:nyc,0:nzc))
allocate(t33(0:nxc,0:nyc,0:nzc))



write(snapID,'(i5)') if1  !index for time snapshot

!Reading the DNS data
filename = 'a_vel_'// trim(adjustl(snapID)) //'.plt' 

open(unit=19, file=filename)
read(19,*) 
read(19,*) 
do k=0,nz            
do j=0,ny
do i=0,nx
read(19,*)x(i,j,k),y(i,j,k),z(i,j,k),u(i,j,k),v(i,j,k),w(i,j,k)
end do
end do
end do
close(19)


filename='AA'// trim(adjustl(corID)) //'_spectrum_DNS_'// trim(adjustl(snapID)) //'.plt' 
call spectrum3d(nx,ny,nz,u,v,w,filename)


!obtain LES field:
!direct coarsening:old way
do k=0,nzc
do j=0,nyc
do i=0,nxc
xc(i,j,k)=x(nr*i,nr*j,nr*k)
yc(i,j,k)=y(nr*i,nr*j,nr*k)
zc(i,j,k)=z(nr*i,nr*j,nr*k)
uc(i,j,k)=u(nr*i,nr*j,nr*k)
vc(i,j,k)=v(nr*i,nr*j,nr*k)
wc(i,j,k)=w(nr*i,nr*j,nr*k)
end do
end do
end do

filename='AA'// trim(adjustl(corID)) //'_spectrum_CRS_'// trim(adjustl(snapID)) //'.plt' 
call spectrum3d(nxc,nyc,nzc,uc,vc,wc,filename)

filename ='AA'// trim(adjustl(corID)) //'_velocity_CRS_'// trim(adjustl(snapID)) //'.plt' 
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z","u","v","w"'
write(19,*) 'zone T=','Zone_ i=', nxc+1, ' j=', nyc+1,',k=',nzc+1,' f=point'
do k=0,nzc
do j=0,nyc
do i=0,nxc
write(19,*)xc(i,j,k),yc(i,j,k),zc(i,j,k),uc(i,j,k),vc(i,j,k),wc(i,j,k)
end do
end do
end do
close(19)

!fft coarsening:correct way
call coarsen(nx,ny,nz,nxc,nyc,nzc,u,uc)
call coarsen(nx,ny,nz,nxc,nyc,nzc,v,vc)
call coarsen(nx,ny,nz,nxc,nyc,nzc,w,wc)

filename='AA'// trim(adjustl(corID)) //'_spectrum_LES_'// trim(adjustl(snapID)) //'.plt' 
call spectrum3d(nxc,nyc,nzc,uc,vc,wc,filename)

filename ='AA'// trim(adjustl(corID)) //'_velocity_LES_'// trim(adjustl(snapID)) //'.plt' 
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z","u","v","w"'
write(19,*) 'zone T=','Zone_ i=', nxc+1, ' j=', nyc+1,',k=',nzc+1,' f=point'
do k=0,nzc
do j=0,nyc
do i=0,nxc
write(19,*)xc(i,j,k),yc(i,j,k),zc(i,j,k),uc(i,j,k),vc(i,j,k),wc(i,j,k)
end do
end do
end do
close(19)


!compute true SGS stress (deviatoric part) on LES grid
call sgs(nx,ny,nz,nxc,nyc,nzc,u,v,w,uc,vc,wc,t11,t12,t13,t22,t23,t33)

filename ='AA'// trim(adjustl(corID)) //'_trueSGS_field_'// trim(adjustl(snapID)) //'.plt' 
open(unit=19, file=filename)
write(19,*) 'variables = "x", "y", "z","t11","t12","t13","t22","t23","t33"'
write(19,*) 'zone T=','Zone_ i=', nxc+1, ' j=', nyc+1,',k=',nzc+1,' f=point'
do k=0,nzc
do j=0,nyc
do i=0,nxc
write(19,*)xc(i,j,k),yc(i,j,k),zc(i,j,k),t11(i,j,k),t12(i,j,k),t13(i,j,k),t22(i,j,k),t23(i,j,k),t33(i,j,k)
end do
end do
end do
close(19)


!Compute PDF of true SGS terms:64 bins
nbin=64

filename = 'AA'// trim(adjustl(corID)) //'_PDF_64bin_trueSGS_t11_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t11,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_64bin_trueSGS_t12_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t12,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_64bin_trueSGS_t13_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t13,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_64bin_trueSGS_t22_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t22,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_64bin_trueSGS_t23_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t23,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_64bin_trueSGS_t33_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t33,filename)

!Compute PDF of true SGS terms:32 bins
nbin=32

filename = 'AA'// trim(adjustl(corID)) //'_PDF_32bin_trueSGS_t11_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t11,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_32bin_trueSGS_t12_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t12,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_32bin_trueSGS_t13_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t13,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_32bin_trueSGS_t22_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t22,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_32bin_trueSGS_t23_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t23,filename)

filename = 'AA'// trim(adjustl(corID)) //'_PDF_32bin_trueSGS_t33_'// trim(adjustl(snapID)) //'.plt' 
call pdf_calc(nxc,nyc,nzc,nbin,t33,filename)


deallocate(x,y,z,xc,yc,zc,u,v,w,uc,vc,wc,t11,t12,t13,t22,t23,t33)  
end do


end 


!-----------------------------------------------------------------!
!Compute coarsening using FFT
!-----------------------------------------------------------------!
subroutine coarsen(nx,ny,nz,nxc,nyc,nzc,u,uc)
implicit none
integer::nx,ny,nz
integer::nxc,nyc,nzc
double precision ::u(0:nx,0:ny,0:nz)
double precision ::uc(0:nxc,0:nyc,0:nzc)
integer::i,j,k,p
double precision,dimension(:),allocatable:: data1d
double precision,dimension(:),allocatable:: data1c
complex*16, dimension(:,:,:),allocatable::gm(:,:,:),gc(:,:,:)
integer,parameter::ndim=3
integer::nn(ndim),isign
double precision ::temp

allocate(data1d(2*nx*ny*nz))
allocate(data1c(2*nxc*nyc*nzc))

!finding fourier coefficients
p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
data1d(p)  =u(i,j,k)
data1d(p+1)=0.0d0
p=p+2
end do
end do
end do

!normalize
temp=1.0d0/dfloat(nx*ny*nz)
do p=1,2*nx*ny*nz
data1d(p)=data1d(p)*temp
end do

!invese fourier transform
nn(1)= nx
nn(2)= ny
nn(3)= nz
isign=-1
call fourn(data1d,nn,ndim,isign)

!in the following: gm=fine data & gc=coarsen data

!finding fourier coefficients: real and imaginary
allocate(gm(0:nx-1,0:ny-1,0:nz-1))
p=1
do k=0,nz-1
do j=0,ny-1
do i=0,nx-1
gm(i,j,k)= dcmplx(data1d(p),data1d(p+1))  
p=p+2
end do
end do
end do

!coarsen:
allocate(gc(0:nxc-1,0:nyc-1,0:nzc-1))

do k=0,nzc/2
	do j=0,nyc/2
		do i=0,nxc/2
		gc(i,j,k) = gm(i,j,k)
		end do
		do i=nxc/2+1,nxc-1
		gc(i,j,k) = gm(i+(nx-nxc),j,k)
		end do
	end do
end do

do k=0,nzc/2
	do j=nyc/2+1,nyc-1
		do i=0,nxc/2
		gc(i,j,k) = gm(i,j+(ny-nyc),k)
		end do
		do i=nxc/2+1,nxc-1
		gc(i,j,k) = gm(i+(nx-nxc),j+(ny-nyc),k)
		end do
	end do
end do

do k=nzc/2+1,nzc-1
	do j=0,nyc/2
		do i=0,nxc/2
		gc(i,j,k) = gm(i,j,k+(nz-nzc))
		end do
		do i=nxc/2+1,nxc-1
		gc(i,j,k) = gm(i+(nx-nxc),j,k+(nz-nzc))
		end do
	end do
end do

do k=nzc/2+1,nzc-1
	do j=nyc/2+1,nyc-1
		do i=0,nxc/2
		gc(i,j,k) = gm(i,j+(ny-nyc),k+(nz-nzc))
		end do
		do i=nxc/2+1,nxc-1
		gc(i,j,k) = gm(i+(nx-nxc),j+(ny-nyc),k+(nz-nzc))
		end do
	end do
end do

!coarsening
p=1
do k=0,nzc-1
do j=0,nyc-1
do i=0,nxc-1
data1c(p)=dreal(gc(i,j,k))
data1c(p+1)=dimag(gc(i,j,k))
p=p+2
end do
end do
end do

!forward fourier transform
nn(1)= nxc
nn(2)= nyc
nn(3)= nzc
isign= 1
call fourn(data1c,nn,ndim,isign)

p=1
do k=0,nzc-1
do j=0,nyc-1
do i=0,nxc-1
uc(i,j,k) = data1c(p)
p=p+2
end do
end do
end do



deallocate(data1d,data1c,gc,gm)

return
end 


!-----------------------------------------------------------------------------------------!
!SGS tensor (deviatoric part)
!-----------------------------------------------------------------------------------------!
subroutine sgs(nx,ny,nz,nxc,nyc,nzc,u,v,w,uc,vc,wc,t11,t12,t13,t22,t23,t33)
implicit none
integer :: nx,ny,nz,nxc,nyc,nzc,i,j,k
double precision,dimension(0:nx,0:ny,0:nz) :: u,v,w
double precision,dimension(0:nxc,0:nyc,0:nzc) :: uc,vc,wc,t11,t12,t13,t22,t23,t33
double precision,dimension(:,:,:),allocatable :: t11a,t12a,t13a,t22a,t23a,t33a,tii

allocate(t11a(0:nx,0:ny,0:nz))
allocate(t12a(0:nx,0:ny,0:nz))
allocate(t13a(0:nx,0:ny,0:nz))
allocate(t22a(0:nx,0:ny,0:nz))
allocate(t23a(0:nx,0:ny,0:nz))
allocate(t33a(0:nx,0:ny,0:nz))

allocate(tii(0:nxc,0:nyc,0:nzc))

!compute nonlinear interatcions on DNS grid
do k = 0,nz
do j = 0,ny
do i = 0,nx
t11a(i,j,k) = u(i,j,k)*u(i,j,k)
t12a(i,j,k) = u(i,j,k)*v(i,j,k)
t13a(i,j,k) = u(i,j,k)*w(i,j,k)
t22a(i,j,k) = v(i,j,k)*v(i,j,k)
t23a(i,j,k) = v(i,j,k)*w(i,j,k)
t33a(i,j,k) = w(i,j,k)*w(i,j,k)
end do
end do
end do

!Coarsen to LES grid:
call coarsen(nx,ny,nz,nxc,nyc,nzc,t11a,t11)
call coarsen(nx,ny,nz,nxc,nyc,nzc,t12a,t12)
call coarsen(nx,ny,nz,nxc,nyc,nzc,t13a,t13)
call coarsen(nx,ny,nz,nxc,nyc,nzc,t22a,t22)
call coarsen(nx,ny,nz,nxc,nyc,nzc,t23a,t23)
call coarsen(nx,ny,nz,nxc,nyc,nzc,t33a,t33)

!Compute SGS:
do k = 0,nzc
do j = 0,nyc
do i = 0,nxc
t11(i,j,k) = uc(i,j,k)*uc(i,j,k)-t11(i,j,k)
t12(i,j,k) = uc(i,j,k)*vc(i,j,k)-t12(i,j,k)
t13(i,j,k) = uc(i,j,k)*wc(i,j,k)-t13(i,j,k)
t22(i,j,k) = vc(i,j,k)*vc(i,j,k)-t22(i,j,k)
t23(i,j,k) = vc(i,j,k)*wc(i,j,k)-t23(i,j,k)
t33(i,j,k) = wc(i,j,k)*wc(i,j,k)-t33(i,j,k)
end do
end do
end do

!compute deviatoric part:
do k = 0,nzc
do j = 0,nyc
do i = 0,nxc
tii(i,j,k) = (t11(i,j,k)+t22(i,j,k)+t33(i,j,k))/3.0d0
end do
end do
end do

do k = 0,nzc
do j = 0,nyc
do i = 0,nxc
t11(i,j,k) = t11(i,j,k) - tii(i,j,k)
t22(i,j,k) = t22(i,j,k) - tii(i,j,k)
t33(i,j,k) = t33(i,j,k) - tii(i,j,k)
end do
end do
end do


deallocate(t11a,t12a,t13a,t22a,t23a,t33a,tii)
return
end



!-----------------------------------------------------------------------------------------!
!PDF Calculator (nx,ny,np,jcff,filename)
!-----------------------------------------------------------------------------------------!
subroutine pdf_calc(nx,ny,nz,np,f,filename)
implicit none
integer :: nx,ny,nz,np,i,j,k,n
character(80) :: filename
real*8,dimension(0:nx,0:ny,0:nz) :: f
real*8 :: mean,sd,s_ij_max,s_ij_min,sum1
real*8,allocatable :: sij(:),fsij(:),psij(:),pdf_sij(:)


!compute mean:
mean = 0.0d0
do k = 1,nz
do j = 1,ny
do i = 1,nx
mean = mean + f(i,j,k)
end do
end do
end do
mean = mean/dfloat(nx*ny*nz)

!standard deviation:
sd = 0.0d0
do k = 1,nz
do j = 1,ny
do i = 1,nx
sd = sd + (f(i,j,k)-mean)**2
end do
end do
end do
sd = sd/dfloat(nx*ny*nz)
sd = dsqrt(sd)

!PDF requirements
s_ij_max = + sd*4.0d0
s_ij_min = - sd*4.0d0

!s_ij_max = min(mean + sd*4.0d0, maxval(f))
!s_ij_min = max(mean - sd*4.0d0, minval(f))



allocate(fsij(0:np))
allocate(sij(1:np))
    
do n = 0,np
  fsij(n) = s_ij_min + (s_ij_max-s_ij_min)*dfloat(n)/dfloat(np)
end do

allocate(pdf_sij(1:np))
do n = 1,np
  pdf_sij(n) = 0.0d0
end do

do n = 1,np
  do k = 0,nz
	do j = 0,ny
  		do i = 0,nx
			if (f(i,j,k).lt.fsij(n).and.f(i,j,k).ge.fsij(n-1)) then
        	pdf_sij(n) = pdf_sij(n)+1.0d0
        	end if
    	end do
  	end do
  end do
  sij(n) = 0.5d0*(fsij(n)+fsij(n-1))
end do

sum1 = 0.0d0
do n = 1,np
  sum1 = sum1 + pdf_sij(n)*(fsij(n)-fsij(n-1))
end do

if (dabs(sum1).le.1.0d-10) sum1 = 1.0d-10

!normalize
allocate(psij(1:np))
do n = 1,np
  psij(n)=pdf_sij(n)/sum1
end do

! writing rhpdf:
open(unit=19, file=filename)
write(19,*) 'variables = "SGS", "PDF", "PDF_max"'
do n=1,np
write(19,*)sij(n),psij(n),psij(n)/maxval(psij)
end do
close(19)

return
end


!-----------------------------------------------------------------!
!Compute spectrum from by definition of Kida
!S Kida, Y. Murakami, et al. 1990
!-----------------------------------------------------------------!
subroutine spectrum3d(nx,ny,nz,u,v,w,filename)
implicit none
integer::nx,ny,nz
character(80) :: filename
real*8 ::u(0:nx,0:ny,0:nz),v(0:nx,0:ny,0:nz),w(0:nx,0:ny,0:nz)
integer::i,j,k,p,np
real*8 ::kx(0:nx),ky(0:ny),kz(0:nz)
real*8,dimension(:),allocatable:: data1d,data2d,data3d,ae
real*8,dimension(:,:,:),allocatable::ee
integer,parameter::ndim=3
integer::nn(ndim),isign
real*8 ::temp,kr

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
open(unit=802, file=filename)
write(802,*)'variables ="k","E(k)"'
do p=1,np
write(802,*)dfloat(p),ae(p)
end do
close(802)

deallocate(data1d,data2d,ee,ae)

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
! compact interpolations for derivarives
!------------------------------------------------------------------!

!------------------------------------------------------------------!
! a second order scheme for first-degree derivative(up)
!------------------------------------------------------------------!
subroutine c2dp(u,up,h,n)
implicit none
integer :: n,i
double precision   :: h!,alpha,beta
double precision , dimension (0:n)  :: u,up
!double precision , dimension (0:n-1):: a,b,c,r,x 

do i=1,n-1
up(i) = (u(i+1)-u(i-1))/(2.0d0*h) 
end do
up(0) = (u(0+1)-u(n-1))/(2.0d0*h) 
up(n) = (u(0+1)-u(n-1))/(2.0d0*h) 

return
end


!------------------------------------------------------------------!
! c4dp:  4th-order compact scheme for first-degree derivative(up)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!
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
! a second order scheme for second-degree derivative(upp)
!------------------------------------------------------------------!
subroutine c2ddp(u,upp,h,n)
implicit none
integer :: n,i
double precision   :: h!,alpha,beta
double precision , dimension (0:n)  :: u,upp
!double precision , dimension (0:n-1):: a,b,c,r,x 

do i=1,n-1
upp(i) = (u(i-1)-2.0d0*u(i)+u(i+1))/(h*h) 
end do
upp(0) = (u(n-1)-2.0d0*u(0)+u(0+1))/(h*h) 
upp(n) = (u(n-1)-2.0d0*u(n)+u(0+1))/(h*h) 

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


