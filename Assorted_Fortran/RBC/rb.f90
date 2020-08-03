!------------------------------------------------------------------------------!
!				     <<< 2D Boussinesq Solver for RBC >>>
!------------------------------------------------------------------------------!
!2D Unsteady Vorticity-Stream Function Formulation
!Ra,Pr are main dimensionless parameters; Re = sqrt(Ra/Pr)
!Rayleigh-Bennard convection
!periodic in left/right
!no-slip/drichlet in top/bottom
!Compact schemes are implemented
!------------------------------------------------------------------------------!
!Omer San
!Oklahoma State University, Stillwater
!CFDLab.org, cfdlab.osu@gmail.com
!Updated: April 28, 2016
!------------------------------------------------------------------------------!
program boussinesq
implicit none
integer::i,j,k,nx,ny,nt,isc,ich,ia,iend,istr,imid,freq,nxi,nyi,ifinal
integer::imv,nfile,ifile,isgs,NA,iadt,idyn
real*8 ::pi,Lx,Ly,Tmax,Tave,Tmid,tt,an,tx,t1,t2,t3,CSMA,betaAD,CSw,CSt,PrT,cpuh
real*8 ::del,cfl,dx,dy,dt,ss,dtfile,tc,tiny
real*8,allocatable::x(:),y(:),ta(:),aa(:),ts(:,:),ts1(:,:),tsNU(:)
real*8,allocatable::s(:,:),w(:,:),t(:,:),tm(:,:),sm(:,:),wm(:,:)
real*8 ::Ra,Pr,Re
real*8 ::TL,TU,afil,kappa2,minNu,maxNu,varNu
real*8 ::NuL,NuT,Ene,Ens,Temp1,Temp2,Temp3,Temp4,Temp5,Nu,Ene0,dis,rate,CSwA,CStA
real*8 ::NuLA,NuTA,EneA,EnsA,Temp1A,Temp2A,Temp3A,Temp4A,Temp5A,NuA,disA,rateA


!Physical Dimensionless Parameters
common /phys/ Ra,Pr,Re

!Filtering paramater
common /Padefilter/ afil

!LES models
common /dynoption/ idyn
common /dyncoeff/ CSw,CSt
common /smacoeff/ CSMA,PrT
common /LESmodels/ isgs
common /ADModel/ betaAD,NA,iadt
common /filratio/ kappa2
     
!read input file
open(10,file='input.txt')
read(10,*)nx
read(10,*)ny
read(10,*)isgs
read(10,*)kappa2
read(10,*)CSMA
read(10,*)PrT
read(10,*)idyn
read(10,*)NA
read(10,*)betaAD
read(10,*)afil
read(10,*)iadt
read(10,*)del
read(10,*)cfl
read(10,*)Lx
read(10,*)Ly
read(10,*)Tmax
read(10,*)Tave
read(10,*)TL
read(10,*)TU
read(10,*)Ra  !Rayleigh number
read(10,*)Pr  !Prandtl number
read(10,*)an
read(10,*)freq
read(10,*)isc
read(10,*)imv
read(10,*)nfile
read(10,*)ifinal
read(10,*)ich
close(10)

!Reynolds number
Re = dsqrt(Ra/Pr)

tiny = 1.0d-8
CSw = CSMA
CSt = CSMA

!check input file
if(ich.ne.19) then
print*,"*** check input file ***"
stop
end if

pi = 4.0d0*datan(1.0d0)


!grid step size
dx=Lx/dfloat(nx)
dy=Ly/dfloat(ny)

!time stepping (initial)
nt=100000000
dt=Tmax/dfloat(nt)
!we will update dt from velocity field using constant CFL condition

!coordinates
allocate(x(0:nx))
allocate(y(0:ny))
do i=0,nx
x(i)=dfloat(i)*dx
end do
do j=0,ny
y(j)=dfloat(j)*dy
end do


!allocate arrays
allocate(s(0:nx,0:ny))
allocate(w(0:nx,0:ny))
allocate(t(0:nx,0:ny))

allocate(tm(0:nx,0:ny))
allocate(sm(0:nx,0:ny))
allocate(wm(0:nx,0:ny))

if (ifinal.eq.0) then !from zero
!initial conditions (resting flow)
tt = 0.0d0
ifile = 0
do j=0,ny
do i=0,nx
s(i,j) = 0.0d0
!w(i,j) = an*dsin(2.0d0*pi*x(i))*dsin(2.0d0*pi*y(j))
w(i,j) = an*dsin(pi*x(i))*dsin(pi*y(j)) !perturbation to initiate stability
!t(i,j) = 0.0d0  !from zero temperature
!t(i,j) = 0.5d0*(TL+TU)  !from constant temperature
t(i,j) = TL+dfloat(j)/dfloat(ny)*(TU-TL)  !from linear profile between upper and lower plates
end do
end do
call psolver(nx,ny,dx,dy,-w,s)
do i=0,nx
	t(i,0) = TL
	t(i,ny)= TU
end do

else ! from 'final.dat' file
  
	open(3,file='final.dat')
	read(3,*)tt
    read(3,*)ifile
    read(3,*)Tave
	read(3,*)nxi,nyi
		if(nxi.ne.nx.or.nyi.ne.ny) then
		write(*,*)'check final.dat file..'
		stop
		end if
	read(3,*)((w(i,j), i=0,nx), j=0,ny)
    read(3,*)((s(i,j), i=0,nx), j=0,ny)
    read(3,*)((t(i,j), i=0,nx), j=0,ny)
	close(3)

end if

!plot initial field
call outfiles(ifile,nx,ny,tt,x,y,s,w,t)

   
call cpu_time(t1)

open(18,file='f_history_rate.plt')
write(18,*) 'variables ="t","rate"'

open(19,file='f_history_Cs.plt')
write(19,*) 'variables ="t","CSw","CSt","PrT"'

!history file (5 sensor points selected to track temperature variation)
open(11,file='f_history.plt')
write(11,*) 'variables ="t","Nu","NuL","NuT","E","W","Dis","T1","T2","T3","T4","T5"'
call history(nx,ny,dx,dy,Lx,Ly,s,w,t,dt,cfl,del,Nu,NuL,NuT,Ene,Ens,dis,Temp1,Temp2,Temp3,Temp4,Temp5)
write(11,77) tt,Nu,NuL,NuT,Ene,Ens,dis,Temp1,Temp2,Temp3,Temp4,Temp5
Ene0=Ene


!initialize time mean values
NuA  = 0.0d0
NuLA = 0.0d0
NuTA = 0.0d0
EneA = 0.0d0
EnsA = 0.0d0 
disA = 0.0d0
rateA =0.0d0
Temp1A=0.0d0
Temp2A=0.0d0
Temp3A=0.0d0
Temp4A=0.0d0
Temp5A=0.0d0
CSwA=0.0d0
CStA=0.0d0

ia = 0

allocate(ts(11,1000000))

!avaraged temperature profile
allocate(ta(0:ny))
do j=0,ny
ta(j)=0.0d0
end do

allocate(aa(0:nx))

iend = 0
imid = 0
istr = 0


Tmid = 0.5d0*(Tmax+Tave)
dtfile = (Tmax-tt)/dfloat(nfile)
tc = tt + dtfile

    do j=0,ny
	do i=0,nx
    tm(i,j) = 0.0d0
    sm(i,j) = 0.0d0
    wm(i,j) = 0.0d0
    end do
    end do
    
!Time integration
do k=1,nt
tt = tt+dt

    
	!solver
	call tvdrk3(nx,ny,dx,dy,dt,s,w,t)
 
    !compute history and time step
    call history(nx,ny,dx,dy,Lx,Ly,s,w,t,dt,cfl,del,Nu,NuL,NuT,Ene,Ens,dis,Temp1,Temp2,Temp3,Temp4,temp5)

    !dissipation rate
    rate = -(Ene-Ene0)/dt
    Ene0 = Ene
	
	!writing history
	if(mod(k,freq).eq.0) then
    write(11,77) tt,Nu,NuL,NuT,Ene,Ens,dis,Temp1,Temp2,Temp3,Temp4,Temp5
    write(19,*) tt,CSw,CSt,(CSw/(CSt+tiny))**2
    write(18,*) tt-0.5d0*dt,rate
	end if

	!write fields for movie
	if (tt.ge.(tc-tiny).and.imv.eq.1) then
    tc = tc + dtfile
    ifile = ifile + 1
	call outfiles(ifile,nx,ny,tt,x,y,s,w,t)
    end if

	!compute statistics
	if(tt.ge.Tave) then

	!compute time avaraged variables: 
	ia= ia + 1
	NuA  = NuA + Nu
	NuLA = NuLA + NuL
	NuTA = NuTA + NuT
	EneA = EneA + Ene
	EnsA = EnsA + Ens 
    disA = disA + dis
    rateA =rateA + rate
	Temp1A=Temp1A + Temp1
	Temp2A=Temp2A + Temp2
	Temp3A=Temp3A + Temp3
	Temp4A=Temp4A + Temp4
	Temp5A=Temp5A + Temp5
	CSwA=CSwA + CSw
	CStA=CStA + CSt

	!time series for pdf
    ts(1,ia)=Nu
    ts(2,ia)=Ene
    ts(3,ia)=Ens
    ts(4,ia)=dis
    ts(5,ia)=Temp1
    ts(6,ia)=Temp2
    ts(7,ia)=Temp3
    ts(8,ia)=Temp4
    ts(9,ia)=Temp5
    ts(10,ia)=CSw
    ts(11,ia)=CSt
    


	do j=0,ny
	do i=0,nx
    tm(i,j) = tm(i,j) + t(i,j)
    sm(i,j) = sm(i,j) + s(i,j)
    wm(i,j) = wm(i,j) + w(i,j)
    end do
    end do
      
	!compute temperature profile (horizontally and time avareged)
	do j=0,ny

		! horizontally averaged temperature	
		do i=0,nx
		aa(i) = t(i,j)
		end do
		call simp1D(nx,dx,aa,ss)
		tx=ss/Lx
		
	ta(j) = ta(j) + tx
	end do

	end if

	!writing starting field
	if (istr.eq.1.and.ifinal.eq.0) then
	istr = 2
	open(101,file='f_start.plt')
	write(101,*)'variables ="x","y","s","w","t"'
	write(101,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',tt,'"'
	do j=0,ny
	do i=0,nx
	write(101,*) x(i),y(j),s(i,j),w(i,j),t(i,j)
	end do
	end do
	close(101)
	end if

	!writing middle field
   	if (imid.eq.1.and.ifinal.eq.0) then
	imid = 2
	open(103,file='f_middle.plt')
	write(103,*)'variables ="x","y","s","w","t"'
	write(103,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',tt,'"'
	do j=0,ny
	do i=0,nx
	write(103,*) x(i),y(j),s(i,j),w(i,j),t(i,j)
	end do
	end do
	close(103)
	end if



	if(isc.eq.1) write(*,'(i10,3es16.6)')k,tt,Ene,dt
	
	if(iend.eq.1) then
	write(11,77) tt,Nu,NuL,NuT,Ene,Ens,dis,Temp1,Temp2,Temp3,Temp4,Temp5
	goto 44
	end if 

	if(istr.eq.0.and.(tt+dt).ge.Tave.and.ifinal.eq.0) then 
	dt = Tave-tt
	istr = 1
	end if

	if(imid.eq.0.and.(tt+dt).ge.Tmid.and.ifinal.eq.0) then 
	dt = Tmid-tt
	imid = 1
	end if


	if((tt+dt).ge.Tmax) then 
	dt = Tmax-tt
	iend = 1
	end if
    
	if(Ene.gt.1.0d10) stop	



	call cpu_time(t3)
	cpuh = (t3-t1)/(60.0d0*60.0d0)
    if (cpuh.ge.503.0d0) goto 44

	
end do

44 continue

close(11)
close(18)
close(19)

call cpu_time(t2)

open(7,file='cpu.txt')
write(7,*) "cpu time  = ", (t2-t1)/(60.0d0*60.0d0), "hours"
close(7)

	!save the data
	open(3,file='final.dat')
	write(3,*)tt
    write(3,*)ifile
    write(3,*)tt
	write(3,*)nx,ny
	write(3,*)((w(i,j), i=0,nx), j=0,ny)
    write(3,*)((s(i,j), i=0,nx), j=0,ny)
    write(3,*)((t(i,j), i=0,nx), j=0,ny)
	close(3)
    

!normalize mean values:
NuA  = NuA/dfloat(ia)
NuLA = NuLA/dfloat(ia)
NuTA = NuTA/dfloat(ia)
EneA = EneA/dfloat(ia)
EnsA = EnsA/dfloat(ia) 
disA = disA/dfloat(ia) 
rateA =rateA/dfloat(ia) 
Temp1A=Temp1A/dfloat(ia)
Temp2A=Temp2A/dfloat(ia)
Temp3A=Temp3A/dfloat(ia)
Temp4A=Temp4A/dfloat(ia)
Temp5A=Temp5A/dfloat(ia)
CSwA=CSwA/dfloat(ia)
CStA=CStA/dfloat(ia)

!normalize temperature profile
do j=0,ny
ta(j)=ta(j)/dfloat(ia)
end do
open(17,file='f_temp_ave.plt')
write(17,*) 'variables ="y","Ta"'
do j=0,ny
write(17,*) y(j),ta(j)
end do
close(17)


	allocate(tsNu(ia))
	do i=1,ia
	tsNu(i)=ts(1,i)
	end do
    minNu = minval(tsNU)
    maxNu = maxval(tsNU)
    !variance
    varNu=0.0d0
  	do i=1,ia
	varNu=varNu + (tsNu(i)-NuA)**2
	end do
    varNu=varNu/dfloat(ia)

open(8,file='f_meanNu.txt')
write(8,'(2es16.6)') ra,NuA
write(8,*) ""
write(8,'(A16,es16.2)') "Ra", ra
write(8,'(A16,f16.6)') "Mean Nu", NuA
write(8,'(A16,f16.6)') "Min  Nu", minNu
write(8,'(A16,f16.6)') "Max  Nu", maxNu
write(8,'(A16,f16.6)') "Var  Nu", varNu
write(8,*) ""
write(8,*) "Numb = ", ia
write(8,*) "Maxt = ", tt
write(8,*) "Avet = ", Tave
write(8,*) ""
write(8,*) "Pr   = ", pr
write(8,*) "Ra   = ", ra
write(8,*) "Re   = ", re
write(8,*) "Nu   = ", NuA
write(8,*) "NuL  = ", NuLA
write(8,*) "NuT  = ", NuTA
write(8,*) "Ene  = ", EneA
write(8,*) "Ens  = ", EnsA
write(8,*) "Dis  = ", disA
write(8,*) "Rate = ", rateA
write(8,*) "Temp1= ", Temp1A
write(8,*) "Temp2= ", Temp2A
write(8,*) "Temp3= ", Temp3A
write(8,*) "Temp4= ", Temp4A
write(8,*) "Temp5= ", Temp5A
write(8,*) "CSw  = ", CSwA
write(8,*) "CSt  = ", CStA

close(8)


!mean fields
	do j=0,ny
	do i=0,nx
    tm(i,j) = tm(i,j)/dfloat(ia)
    sm(i,j) = sm(i,j)/dfloat(ia)
    wm(i,j) = wm(i,j)/dfloat(ia)
    end do
    end do
    
open(107,file='f_mean_field.plt')
write(107,*)'variables ="x","y","s","w","t"'
write(107,*)'zone f=point i=',nx+1,',j=',ny+1
do j=0,ny
do i=0,nx
write(107,*) x(i),y(j),sm(i,j),wm(i,j),tm(i,j)
end do
end do
close(107)


open(102,file='f_final.plt')
write(102,*)'variables ="x","y","s","w","t"'
write(102,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',tt,'"'
do j=0,ny
do i=0,nx
write(102,*) x(i),y(j),s(i,j),w(i,j),t(i,j)
end do
end do
close(102)


!compute probability density function
allocate(ts1(11,ia))
do k=1,11
	do i=1,ia
	ts1(k,i)=ts(k,i)
	end do
end do
call compdf1000(ia,ts1)
call compdf200(ia,ts1)

deallocate(x,y,aa,ta,w,s,t,tm,ts,ts1)

51 format(a16,i8,a4,i8,a10,f10.4,a3)
77 format(11es16.6)
end


!-------------------------------------------------------------------------!
! compute probability density function of the time series (normalized)
!-------------------------------------------------------------------------!
subroutine compdf200(ia,ts)
implicit none
integer:: ia,i,j,k,npdf
real*8 :: ts(11,ia)
real*8 :: tn(ia),intg,dp,minP,maxP
real*8,dimension(:),allocatable::p
integer,dimension(:),allocatable::np


npdf=min(ia,200)

allocate(p(npdf))
allocate(np(npdf))

open(11,file='f_pfd_200.plt')
write(11,*) 'variables ="p","pdf"'
    
do k=1,11
	do i=1,ia
	tn(i)=ts(k,i)
	end do


	!compute pdf for power
	minP = minval(tn)
	maxP = maxval(tn)

	dp = (maxP-minP)/dfloat(npdf)

	do j=1,npdf
   	p(j) = minP + dfloat(j)*dp 
        	
    np(j)  = 0
    	do i=1,ia
        
    		if (tn(i).ge.p(j)-dp/2.0d0.and.tn(i).le.p(j)+dp/2.0d0) then
                
			np(j) = np(j) + 1
         
            end if	

        end do
      		
	end do
   
	!scaling to the unity
	intg = 0.0d0
	do j=1,npdf
	intg = intg + dfloat(np(j))*dp
	end do

    !check for steady case
    if (dabs(intg).le.1.0d-8) intg = 1.0d-8
      
	write(11,*)'zone f=point i=',npdf,',t="case',k,'"'
	do j=1,npdf
	write(11,*)p(j),np(j)/intg
	end do
	
end do

close(11)

return
end


!-------------------------------------------------------------------------!
! compute probability density function of the time series (normalized)
!-------------------------------------------------------------------------!
subroutine compdf1000(ia,ts)
implicit none
integer:: ia,i,j,k,npdf
real*8 :: ts(11,ia)
real*8 :: tn(ia),intg,dp,minP,maxP
real*8,dimension(:),allocatable::p
integer,dimension(:),allocatable::np


npdf=min(ia,1000)

allocate(p(npdf))
allocate(np(npdf))

open(11,file='f_pfd_1000.plt')
write(11,*) 'variables ="p","pdf"'
    
do k=1,11
	do i=1,ia
	tn(i)=ts(k,i)
	end do


	!compute pdf for power
	minP = minval(tn)
	maxP = maxval(tn)

	dp = (maxP-minP)/dfloat(npdf)

	do j=1,npdf
   	p(j) = minP + dfloat(j)*dp 
        	
    np(j)  = 0
    	do i=1,ia
        
    		if (tn(i).ge.p(j)-dp/2.0d0.and.tn(i).le.p(j)+dp/2.0d0) then
                
			np(j) = np(j) + 1
         
            end if	

        end do
      		
	end do
   
	!scaling to the unity
	intg = 0.0d0
	do j=1,npdf
	intg = intg + dfloat(np(j))*dp
	end do

    !check for steady case
    if (dabs(intg).le.1.0d-8) intg = 1.0d-8
      
	write(11,*)'zone f=point i=',npdf,',t="case',k,'"'
	do j=1,npdf
	write(11,*)p(j),np(j)/intg
	end do
	
end do

close(11)

return
end


!-------------------------------------------------------------------------!
! compute smagorinsky eddy viscosity model
!-------------------------------------------------------------------------!
subroutine smag(nx,ny,dx,dy,s,w,t,gw,gt)
implicit none
integer:: nx,ny
real*8 :: dx,dy,CSMA,PrT
real*8 :: w(0:nx,0:ny),s(0:nx,0:ny),t(0:nx,0:ny),gw(0:nx,0:ny),gt(0:nx,0:ny)
integer:: i,j
real*8, allocatable :: mw(:,:),mt(:,:)

common /smacoeff/ CSMA,PrT

allocate(mw(0:nx,0:ny))
allocate(mt(0:nx,0:ny))

call evM(nx,ny,dx,dy,s,w,mw)
call evM(nx,ny,dx,dy,s,t,mt)

	do j=0,ny
	do i=0,nx
		gw(i,j) = CSMA*CSMA*dx*dy*mw(i,j)
        gt(i,j) = CSMA*CSMA*dx*dy*mt(i,j)/PrT
	end do
	end do

deallocate(mw,mt)

return
end

!-------------------------------------------------------------------------!
! compute dynamic eddy viscosity model (new model)
!-------------------------------------------------------------------------!
subroutine dyn_new(nx,ny,dx,dy,s,w,t,gw,gt)
implicit none
integer:: nx,ny
real*8 :: dx,dy,ddw,nnw,csdw,ddt,nnt,csdt,CSw,CSt,CSddw,CSddt
real*8 :: w(0:nx,0:ny),s(0:nx,0:ny),t(0:nx,0:ny),gw(0:nx,0:ny),gt(0:nx,0:ny)
real*8, dimension (:,:), allocatable :: sa,wa,ta,jw,jt,jwa,jta,jwaf,jtaf
real*8, dimension (:,:), allocatable :: llw,llt,mmw,mmt
integer:: i,j,idyn
real*8::csdhw(0:ny),csdht(0:ny)

common /dyncoeff/ CSw,CSt
common /dynoption/ idyn

allocate(llw(0:nx,0:ny))
allocate(llt(0:nx,0:ny))
allocate(mmw(0:nx,0:ny))
allocate(mmt(0:nx,0:ny))

!Compute nonlinear Jacobian
allocate(jw(0:nx,0:ny))
allocate(jt(0:nx,0:ny))
call jacobians(nx,ny,dx,dy,s,w,t,jw,jt)  

!AD process
allocate(sa(0:nx,0:ny))
allocate(wa(0:nx,0:ny))
allocate(ta(0:nx,0:ny))
	
call adm(nx,ny,s,sa)
call adm(nx,ny,w,wa)
call adm(nx,ny,t,ta)

	!Compute nonlinear Jacobian 
	allocate(jwa(0:nx,0:ny))
    allocate(jta(0:nx,0:ny))
    call jacobians(nx,ny,dx,dy,sa,wa,ta,jwa,jta)  
	
    !filter
    allocate(jwaf(0:nx,0:ny))
    allocate(jtaf(0:nx,0:ny))
    call filter(nx,ny,jwa,jwaf)
    call filter(nx,ny,jta,jtaf)
    
	!compute L
	do j=0,ny
	do i=0,nx
	llw(i,j)=jw(i,j) - jwaf(i,j)
    llt(i,j)=jt(i,j) - jtaf(i,j)
	end do
	end do

    !Compute M
	call evM(nx,ny,dx,dy,s,w,mmw)
    call evM(nx,ny,dx,dy,s,t,mmt)
  

!compute (cs*delta)^2 =csd 	
if (idyn.eq.2) then  !abs value
  
	nnw = 0.0d0
	ddw = 0.0d0
    nnt = 0.0d0
	ddt = 0.0d0	
	do j=0,ny
	do i=0,nx 
	nnw = nnw + dabs(llw(i,j)*mmw(i,j))
	ddw = ddw + mmw(i,j)*mmw(i,j)
    nnt = nnt + dabs(llt(i,j)*mmt(i,j))
	ddt = ddt + mmt(i,j)*mmt(i,j)
	end do
	end do

    !compute csd
	csdw = dabs(nnw/ddw)
	csdt = dabs(nnt/ddt)

    
	CSw = dsqrt(csdw/(dx*dy))
    CSt = dsqrt(csdt/(dx*dy))
    
	do j=0,ny
	do i=0,nx
		gw(i,j) = csdw*mmw(i,j)
        gt(i,j) = csdt*mmt(i,j)
	end do
	end do

    
else if (idyn.eq.3) then  !local

	nnw = 0.0d0
	ddw = 0.0d0
    nnt = 0.0d0
	ddt = 0.0d0	
	do j=0,ny
	do i=0,nx 
	nnw = nnw + llw(i,j)*mmw(i,j)
	ddw = ddw + mmw(i,j)*mmw(i,j)
    nnt = nnt + llt(i,j)*mmt(i,j)
	ddt = ddt + mmt(i,j)*mmt(i,j)
	end do
	end do

    !compute csd
	csdw = dabs(nnw/ddw)
	csdt = dabs(nnt/ddt)

    
	CSw = dsqrt(csdw/(dx*dy))
    CSt = dsqrt(csdt/(dx*dy))
    
	do j=0,ny
	do i=0,nx
		gw(i,j) = csdw*mmw(i,j)
        gt(i,j) = csdt*mmt(i,j)
	end do
	end do

else !homogeneous (I think that is more physical)

	CSddw = 0.0d0 !only for printing purpose
    CSddt = 0.0d0 !only for printing purpose

    !x is the homogeneous direction
    !perform averaging only for x-direction
	do j=0,ny
    
    	nnw = 0.0d0
		ddw = 0.0d0
    	nnt = 0.0d0
		ddt = 0.0d0
		do i=0,nx 
		nnw = nnw + llw(i,j)*mmw(i,j)
		ddw = ddw + mmw(i,j)*mmw(i,j)
    	nnt = nnt + llt(i,j)*mmt(i,j)
		ddt = ddt + mmt(i,j)*mmt(i,j)
		end do
        
		!compute csd (which is function of y)
		csdhw(j) = dabs(nnw/ddw)
		csdht(j) = dabs(nnt/ddt)

        CSddw = CSddw + csdhw(j)  !only for printing purpose
        CSddt = CSddt + csdht(j)  !only for printing purpose
        
	end do
		
	
  	do j=0,ny
	do i=0,nx
		gw(i,j) = csdhw(j)*mmw(i,j)
        gt(i,j) = csdht(j)*mmt(i,j)
	end do
	end do  
  
	!compute for printing
    CSddw = CSddw/dfloat(ny+1)
    CSddt = CSddt/dfloat(ny+1)
	CSw = dsqrt(CSddw/(dx*dy))
    CSt = dsqrt(CSddt/(dx*dy))
  
end if
  
deallocate(sa,wa,ta,jw,jt,jwa,jta,jwaf,jtaf,llw,llt,mmw,mmt)


return
end



!-------------------------------------------------------------------------!
! compute dynamic eddy viscosity model (classical model)
!-------------------------------------------------------------------------!
subroutine dyn_std(nx,ny,dx,dy,s,w,t,gw,gt)
implicit none
integer:: nx,ny
real*8 :: dx,dy,ddw,nnw,csdw,ddt,nnt,csdt,CSw,CSt,kappa2,CSddw,CSddt
real*8 :: w(0:nx,0:ny),s(0:nx,0:ny),t(0:nx,0:ny),gw(0:nx,0:ny),gt(0:nx,0:ny)
real*8, dimension (:,:), allocatable :: sf,wf,tf,fjw,fjt,jw,jt,jwf,jtf
real*8, dimension (:,:), allocatable :: llw,llt,mmw,mmt,mmwf,mmtf,mmwr,mmtr
integer:: i,j,idyn
real*8::csdhw(0:ny),csdht(0:ny)


common /filratio/ kappa2
common /dyncoeff/ CSw,CSt
common /dynoption/ idyn

	!compute jacobian of filtered variables
	allocate(wf(0:nx,0:ny))
	allocate(tf(0:nx,0:ny))
    allocate(sf(0:nx,0:ny))
	
	call filter(nx,ny,w,wf)
    call filter(nx,ny,t,tf)
	call filter(nx,ny,s,sf)

	allocate(fjw(0:nx,0:ny))
    allocate(fjt(0:nx,0:ny))
	call jacobians(nx,ny,dx,dy,sf,wf,tf,fjw,fjt)
    
	
    !Compute nonlinear Jacobian
	allocate(jw(0:nx,0:ny))
	allocate(jt(0:nx,0:ny))
	call jacobians(nx,ny,dx,dy,s,w,t,jw,jt)
    
	!compute filtered jacobian
	allocate(jwf(0:nx,0:ny))
    allocate(jtf(0:nx,0:ny))
	call filter(nx,ny,jw,jwf)
    call filter(nx,ny,jt,jtf)

	allocate(mmw(0:nx,0:ny))
    allocate(mmt(0:nx,0:ny))
  	allocate(mmwf(0:nx,0:ny))
    allocate(mmtf(0:nx,0:ny))
    allocate(mmwr(0:nx,0:ny))
    allocate(mmtr(0:nx,0:ny))
    
    !Compute M from filtered field
	call evM(nx,ny,dx,dy,sf,wf,mmwf)
    call evM(nx,ny,dx,dy,sf,tf,mmtf)

	!Compute M from resolved field
 	call evM(nx,ny,dx,dy,s,w,mmwr)
    call evM(nx,ny,dx,dy,s,t,mmtr)   

    !and filter it
    call filter(nx,ny,mmwr,mmw)
    call filter(nx,ny,mmtr,mmt)
  
	allocate(llw(0:nx,0:ny))
    allocate(llt(0:nx,0:ny))
	!compute L
	do j=0,ny
	do i=0,nx
	llw(i,j)=fjw(i,j) - jwf(i,j)
    llt(i,j)=fjt(i,j) - jtf(i,j)
	end do
	end do

    !compute M
	do j=0,ny
	do i=0,nx
	mmw(i,j)=kappa2*mmwf(i,j) - mmw(i,j)
    mmt(i,j)=kappa2*mmtf(i,j) - mmt(i,j)
	end do
	end do

    
!compute (cs*delta)^2 =csd 
	
if (idyn.eq.2) then  !abs value
	nnw = 0.0d0
	ddw = 0.0d0
    nnt = 0.0d0
	ddt = 0.0d0	
	do j=0,ny
	do i=0,nx 
	nnw = nnw + dabs(llw(i,j)*mmw(i,j))
	ddw = ddw + mmw(i,j)*mmw(i,j)
    nnt = nnt + dabs(llt(i,j)*mmt(i,j))
	ddt = ddt + mmt(i,j)*mmt(i,j)
	end do
	end do

    !compute csd
	csdw = dabs(nnw/ddw)
	csdt = dabs(nnt/ddt)

    
	CSw = dsqrt(csdw/(dx*dy))
    CSt = dsqrt(csdt/(dx*dy))
    
	do j=0,ny
	do i=0,nx
		gw(i,j) = csdw*mmw(i,j)
        gt(i,j) = csdt*mmt(i,j)
	end do
	end do

    
else if (idyn.eq.3) then  !local

	nnw = 0.0d0
	ddw = 0.0d0
    nnt = 0.0d0
	ddt = 0.0d0	
	do j=0,ny
	do i=0,nx 
	nnw = nnw + llw(i,j)*mmw(i,j)
	ddw = ddw + mmw(i,j)*mmw(i,j)
    nnt = nnt + llt(i,j)*mmt(i,j)
	ddt = ddt + mmt(i,j)*mmt(i,j)
	end do
	end do

    !compute csd
	csdw = dabs(nnw/ddw)
	csdt = dabs(nnt/ddt)

    
	CSw = dsqrt(csdw/(dx*dy))
    CSt = dsqrt(csdt/(dx*dy))
    
	do j=0,ny
	do i=0,nx
		gw(i,j) = csdw*mmw(i,j)
        gt(i,j) = csdt*mmt(i,j)
	end do
	end do

else !homogeneous (I think that is more physical)

	CSddw = 0.0d0 !only for printing purpose
    CSddt = 0.0d0 !only for printing purpose

    !x is the homogeneous direction
    !perform averaging only for x-direction
	do j=0,ny
    
    	nnw = 0.0d0
		ddw = 0.0d0
    	nnt = 0.0d0
		ddt = 0.0d0
		do i=0,nx 
		nnw = nnw + llw(i,j)*mmw(i,j)
		ddw = ddw + mmw(i,j)*mmw(i,j)
    	nnt = nnt + llt(i,j)*mmt(i,j)
		ddt = ddt + mmt(i,j)*mmt(i,j)
		end do
        
		!compute csd (which is function of y)
		csdhw(j) = dabs(nnw/ddw)
		csdht(j) = dabs(nnt/ddt)

        CSddw = CSddw + csdhw(j)  !only for printing purpose
        CSddt = CSddt + csdht(j)  !only for printing purpose
        
	end do
		
	
  	do j=0,ny
	do i=0,nx
		gw(i,j) = csdhw(j)*mmw(i,j)
        gt(i,j) = csdht(j)*mmt(i,j)
	end do
	end do  
  
	!compute for printing
    CSddw = CSddw/dfloat(ny+1)
    CSddt = CSddt/dfloat(ny+1)
	CSw = dsqrt(CSddw/(dx*dy))
    CSt = dsqrt(CSddt/(dx*dy))
  
end if
	

deallocate(sf,wf,tf,fjw,fjt,jw,jt,jwf,jtf,llw,llt,mmw,mmt,mmwf,mmtf,mmwr,mmtr)


return
end


!-------------------------------------------------------------------------!
! compute AD-LES model
!-------------------------------------------------------------------------!
subroutine adles(nx,ny,dx,dy,s,w,t,gw,gt)
implicit none
integer:: nx,ny
real*8 :: dx,dy
real*8 :: w(0:nx,0:ny),s(0:nx,0:ny),t(0:nx,0:ny),gw(0:nx,0:ny),gt(0:nx,0:ny)
real*8, dimension (:,:), allocatable :: sa,wa,ta,jw,jt,jwa,jta,jwaf,jtaf
integer:: i,j


!Compute nonlinear Jacobian
allocate(jw(0:nx,0:ny))
allocate(jt(0:nx,0:ny))
call jacobians(nx,ny,dx,dy,s,w,t,jw,jt)  

!AD process
allocate(sa(0:nx,0:ny))
allocate(wa(0:nx,0:ny))
allocate(ta(0:nx,0:ny))
	
call adm(nx,ny,s,sa)
call adm(nx,ny,w,wa)
call adm(nx,ny,t,ta)

	!Compute nonlinear Jacobian
	allocate(jwa(0:nx,0:ny))
    allocate(jta(0:nx,0:ny))
    call jacobians(nx,ny,dx,dy,sa,wa,ta,jwa,jta)  
	
    !filter
    allocate(jwaf(0:nx,0:ny))
    allocate(jtaf(0:nx,0:ny))
    call filter(nx,ny,jwa,jwaf)
    call filter(nx,ny,jta,jtaf)
    
	!compute L
	do j=0,ny
	do i=0,nx
	gw(i,j)=jw(i,j) - jwaf(i,j)
    gt(i,j)=jt(i,j) - jtaf(i,j)
	end do
	end do
  
deallocate(sa,wa,ta,jw,jt,jwa,jta,jwaf,jtaf)


return
end

!-------------------------------------------------------------------------!
! compute M for eddy viscosity LES: div(|S|grad(w))
!-------------------------------------------------------------------------!
subroutine evM(nx,ny,dx,dy,s,w,m)
implicit none
integer:: nx,ny
real*8 :: dx,dy
real*8 :: s(0:nx,0:ny),w(0:nx,0:ny),m(0:nx,0:ny)
integer:: i,j
real*8, dimension (:), allocatable   :: a,b
real*8, dimension (:,:), allocatable :: st,wx,wy

allocate(st(0:nx,0:ny))
call strain(nx,ny,dx,dy,s,st)

allocate(wx(0:nx,0:ny))
allocate(wy(0:nx,0:ny))

! wx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = w(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	wx(i,j) = b(i)
	end do
end do
deallocate(a,b)

! (S.wx)x
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = wx(i,j)*st(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	m(i,j) = b(i)
	end do
end do
deallocate(a,b)


! wy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = w(i,j)
	end do
		call c4d(a,b,dy,ny)
	do j=0,ny
	wy(i,j) = b(j)
	end do
end do
deallocate(a,b)

! (S.wy)y
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = wy(i,j)*st(i,j)
	end do
		call c4d(a,b,dy,ny)
	do j=0,ny
	m(i,j) = m(i,j) + b(j)
	end do
end do
deallocate(a,b)

deallocate(st,wx,wy)

return
end

!-----------------------------------------------------------------!
!Compute strain from stream function
!-----------------------------------------------------------------!
subroutine strain(nx,ny,dx,dy,s,st)
implicit none
integer::nx,ny
real*8 ::dx,dy
real*8 ::s(0:nx,0:ny),st(0:nx,0:ny)
integer::i,j
real*8, dimension (:), allocatable   :: a,b
real*8, dimension (:,:), allocatable :: g,syx,sxx,syy

allocate(g(0:nx,0:ny))
allocate(syx(0:nx,0:ny))
allocate(sxx(0:nx,0:ny))
allocate(syy(0:nx,0:ny))

! sy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4d(a,b,dy,ny)
	do j=0,ny
	g(i,j) = b(j)
	end do
end do
deallocate(a,b)

! syx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = g(i,j)
	end do
		call c4dp(a,b,dx,nx)
	do i=0,nx
	syx(i,j) = b(i)
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
	sxx(i,j) = b(i)
	end do
end do
deallocate(a,b)

! syy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4dd(a,b,dy,ny)
	do j=0,ny
	syy(i,j) = b(j)
	end do
end do
deallocate(a,b)

!strain field
do j=0,ny
do i=0,nx
st(i,j) = dsqrt(4.0d0*syx(i,j)*syx(i,j) + (sxx(i,j)-syy(i,j))*(sxx(i,j)-syy(i,j)))
end do
end do

deallocate(syx,sxx,syy,g)
return
end 


!---------------------------------------------------------------------------!
!Output files
!---------------------------------------------------------------------------!
subroutine outfiles(ifile,nx,ny,tt,x,y,s,w,t)
implicit none
integer::nx,ny,ifile
real*8 ::x(0:nx),y(0:ny)
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),t(0:nx,0:ny)
integer::i,j
real*8 ::tt

open(5000+ifile)
write(5000+ifile,*) 'variables ="x","y","s","w","t"'
write(5000+ifile,51)'zone f=point i=',nx+1,',j=',ny+1,',t="time',tt,'"'
do j=0,ny
do i=0,nx
write(5000+ifile,52) x(i),y(j),s(i,j),w(i,j),t(i,j)
end do
end do
close(5000+ifile)

51 format(a16,i8,a4,i8,a10,f10.4,a3)
52 format(5es18.8)

return
end

!---------------------------------------------------------------------------!
!Compute temporal history and time step 
!---------------------------------------------------------------------------!
subroutine history(nx,ny,dx,dy,Lx,Ly,s,w,t,dt,cfl,del,Nu,NuL,NuT,Ene,Ens,dis,T1,T2,T3,T4,T5)
implicit none
integer::nx,ny
real*8 ::dx,dy,Lx,Ly,ss,cfl,del,dt
real*8 ::umax,umin,vmax,vmin,alpha,dif_in,dt_cfl,dt_del
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),t(0:nx,0:ny)
real*8 ::NuL,NuT,Ene,Ens,T1,T2,T3,T4,T5,Nu,dis
real*8 ::Ra,Pr,Re
integer::i,j
real*8, dimension (:), allocatable   :: a,b
real*8, dimension (:,:), allocatable :: u,v,g

common /phys/ Ra,Pr,Re



!compute velocity components:
allocate(u(0:nx,0:ny),v(0:nx,0:ny))
! u = sy
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


! v = - sx
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

dt_cfl = min(cfl*dx/umax,cfl*dy/vmax)

alpha  = 1.0d0/(dx*dx) + 1.0d0/(dy*dy) 

dif_in = min(re,re*pr)

dt_del = del*dif_in/alpha


dt = min(dt_cfl,dt_del)


!compute total energy:
allocate(g(0:nx,0:ny))
do i=0,nx
do j=0,ny
g(i,j)=0.5d0*(u(i,j)**2 + v(i,j)**2)
end do
end do
call simp2D(nx,ny,dx,dy,g,ss)
deallocate(g)
Ene=ss/(Lx*Ly)



!compute total enstrophy:
allocate(g(0:nx,0:ny))
do i=0,nx
do j=0,ny
g(i,j)=0.5d0*(w(i,j)**2)
end do
end do
call simp2D(nx,ny,dx,dy,g,ss)
deallocate(g)
Ens=ss/(Lx*Ly)

!dissipation
dis = (2.0d0/Re)*Ens

!compute Nusselt numbers

! ty
allocate(g(0:nx,0:ny))
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = t(i,j)
	end do
		call c4d(a,b,dy,ny)
	do j=0,ny
	g(i,j) = b(j)
	end do
end do
deallocate(a,b)

! lower boundary
allocate(a(0:nx))
do i=0,nx
a(i) = g(i,0)
end do
call simp1D(nx,dx,a,ss)
deallocate(a)
NuL=ss/Lx

! top boundary
allocate(a(0:nx))
do i=0,nx
a(i) = g(i,ny)
end do
call simp1D(nx,dx,a,ss)
deallocate(a)
NuT=ss/Lx
deallocate(g)

!mean nusselt number
allocate(g(0:nx,0:ny))
do i=0,nx
do j=0,ny
g(i,j)=v(i,j)*t(i,j)
end do
end do
call simp2D(nx,ny,dx,dy,g,ss)
deallocate(g)
Nu=1.0d0 + dsqrt(ra*pr)*ss/(Lx*Ly)
deallocate(u,v)

T1 = t(nx/8,ny/8)
T2 = t(nx/4,ny/4*3)
T3 = t(nx/2,ny/2)
T4 = t(nx/4*3,ny/4)
T5 = t(nx/8*7,ny/8*7)


return
end


!---------------------------------------------------------------------------!
!TVD Runge-Kutta 3rd-order constant time step integration
!---------------------------------------------------------------------------!
subroutine tvdrk3(nx,ny,dx,dy,dt,s,w,t)
implicit none
integer::nx,ny
real*8 ::dx,dy,dt,a,b,c
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),t(0:nx,0:ny)
integer::i,j
real*8,dimension(:,:),allocatable::rw,rt,ww,tt


allocate(rw(0:nx,0:ny))
allocate(rt(0:nx,0:ny))
allocate(ww(0:nx,0:ny))
allocate(tt(0:nx,0:ny))

!Boundary conditions for vorticity
!bottom-up
do i=0,nx
	ww(i,0) = -1.0d0/(18.0d0*dy*dy)*(108.0d0*s(i,1) - 27.0d0*s(i,2) + 4.0d0*s(i,3))
	ww(i,ny)= -1.0d0/(18.0d0*dy*dy)*(108.0d0*s(i,ny-1) - 27.0d0*s(i,ny-2) + 4.0d0*s(i,ny-3)) 
	tt(i,0) = t(i,0) 
	tt(i,ny) = t(i,ny)
end do

!compute rhs terms:
call rhs(nx,ny,dx,dy,s,w,t,rw,rt)

!first update
do j=1,ny-1
do i=0,nx
tt(i,j) = t(i,j) + dt*rt(i,j)
ww(i,j) = w(i,j) + dt*rw(i,j)
end do
end do


!Elliptic Poisson solver:
call psolver(nx,ny,dx,dy,-ww,s)

!Boundary conditions for vorticity
!bottom-up
do i=0,nx
	ww(i,0) = -1.0d0/(18.0d0*dy*dy)*(108.0d0*s(i,1) - 27.0d0*s(i,2) + 4.0d0*s(i,3))
	ww(i,ny)= -1.0d0/(18.0d0*dy*dy)*(108.0d0*s(i,ny-1) - 27.0d0*s(i,ny-2) + 4.0d0*s(i,ny-3)) 
	tt(i,0) = t(i,0) 
	tt(i,ny) = t(i,ny)
end do


!compute rhs terms:
call rhs(nx,ny,dx,dy,s,ww,tt,rw,rt)

!second update
a = 3.0d0/4.0d0
do j=1,ny-1
do i=0,nx
tt(i,j) = a*t(i,j) + 0.25d0*tt(i,j) + 0.25d0*dt*rt(i,j)
ww(i,j) = a*w(i,j) + 0.25d0*ww(i,j) + 0.25d0*dt*rw(i,j)
end do
end do

!Elliptic Poisson solver:
call psolver(nx,ny,dx,dy,-ww,s)

!Boundary conditions for vorticity
!bottom-up
do i=0,nx
	w(i,0) = -1.0d0/(18.0d0*dy*dy)*(108.0d0*s(i,1) - 27.0d0*s(i,2) + 4.0d0*s(i,3))
	w(i,ny)= -1.0d0/(18.0d0*dy*dy)*(108.0d0*s(i,ny-1) - 27.0d0*s(i,ny-2) + 4.0d0*s(i,ny-3)) 
end do

!compute rhs terms:
call rhs(nx,ny,dx,dy,s,ww,tt,rw,rt)


!third update
b = 1.0d0/3.0d0
c = 2.0d0/3.0d0
do j=1,ny-1
do i=0,nx
t(i,j) = b*t(i,j) + c*tt(i,j) + c*dt*rt(i,j)
w(i,j) = b*w(i,j) + c*ww(i,j) + c*dt*rw(i,j)
end do
end do

!Elliptic Poisson solver:
call psolver(nx,ny,dx,dy,-w,s)

deallocate(rw,rt,ww,tt)

return
end


!---------------------------------------------------------------------------!
!Computing nonlienar terms (Jacobians) at RHS
!---------------------------------------------------------------------------!
subroutine jacobians(nx,ny,dx,dy,s,w,t,jw,jt)
implicit none
integer::nx,ny
real*8 ::dx,dy
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),t(0:nx,0:ny)
real*8 ::jw(0:nx,0:ny),jt(0:nx,0:ny)
integer::i,j
real*8, dimension (:), allocatable   :: a,b
real*8, dimension (:,:), allocatable :: e

! convective term:
allocate(e(0:nx,0:ny))

! sy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4d(a,b,dy,ny)
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
	jw(i,j) = e(i,j)*b(i)
	end do
end do
deallocate(a,b)

! tx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = t(i,j)
	end do
        call c4dp(a,b,dx,nx)
	do i=0,nx
	jt(i,j) = e(i,j)*b(i)
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
        call c4d(a,b,dy,ny)
	do j=0,ny
	jw(i,j) = jw(i,j) - e(i,j)*b(j)
	end do
end do
deallocate(a,b)

! ty
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = t(i,j)
	end do
        call c4d(a,b,dy,ny)
	do j=0,ny
	jt(i,j) = jt(i,j) - e(i,j)*b(j)
	end do
end do
deallocate(a,b)
    
deallocate(e)


return
end

!---------------------------------------------------------------------------!
!Computing rhs of equations
!---------------------------------------------------------------------------!
subroutine rhs(nx,ny,dx,dy,s,w,t,rw,rt)
implicit none
integer::nx,ny
real*8 ::dx,dy
real*8 ::s(0:nx,0:ny),w(0:nx,0:ny),t(0:nx,0:ny)
real*8 ::rw(0:nx,0:ny),rt(0:nx,0:ny)
integer::i,j,isgs
real*8 ::Ra,Pr,Re
real*8, dimension (:), allocatable   :: a,b
real*8, dimension (:,:), allocatable :: e,gw,gt

common /phys/ Ra,Pr,Re
common /LESmodels/ isgs

! viscous terms for vorticity transport equation:
! wxx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = w(i,j)
	end do 
        call c4ddp(a,b,dx,nx)     
	do i=0,nx
	rw(i,j) = b(i)/Re		  
	end do
end do
deallocate(a,b)

! wyy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = w(i,j)
	end do
        call c4dd(a,b,dy,ny)      
	do j=0,ny
	rw(i,j) = rw(i,j) + b(j)/Re
	end do
end do
deallocate(a,b)


! viscous terms for temperature transport equation:
! txx 
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = t(i,j)
	end do 
        call c4ddp(a,b,dx,nx)     
	do i=0,nx
	rt(i,j) = b(i)/(Re*Pr)	   !1/(re*pr)= 1/sqrt(pr*ra)
	end do
end do
deallocate(a,b)

! tyy 
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = t(i,j)
	end do
        call c4dd(a,b,dy,ny)      
	do j=0,ny
	rt(i,j) = rt(i,j) + b(j)/(Re*Pr)
	end do
end do
deallocate(a,b)


! convective term:
allocate(e(0:nx,0:ny))

! sy
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = s(i,j)
	end do
		call c4d(a,b,dy,ny)
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
	rw(i,j) = rw(i,j) - e(i,j)*b(i)
	end do
end do
deallocate(a,b)

! tx
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = t(i,j)
	end do
        call c4dp(a,b,dx,nx)
	do i=0,nx
	rt(i,j) = rt(i,j) - e(i,j)*b(i)
    rw(i,j) = rw(i,j) + b(i)
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
        call c4d(a,b,dy,ny)
	do j=0,ny
	rw(i,j) = rw(i,j) + e(i,j)*b(j)
	end do
end do
deallocate(a,b)

! ty
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = t(i,j)
	end do
        call c4d(a,b,dy,ny)
	do j=0,ny
	rt(i,j) = rt(i,j) + e(i,j)*b(j)
	end do
end do
deallocate(a,b)
    
deallocate(e)


!Add LES models:
if (isgs.eq.2) then !smagorinsky eddy viscosity model

	allocate (gw(0:nx,0:ny))
    allocate (gt(0:nx,0:ny))
	call smag(nx,ny,dx,dy,s,w,t,gw,gt)
    
	do j=0,ny
	do i=0,nx
	rw(i,j) = rw(i,j) + gw(i,j)
    rt(i,j) = rt(i,j) + gt(i,j)    		
	end do
	end do
    
    deallocate(gw,gt)

else if (isgs.eq.3) then !AD-LES model

	allocate (gw(0:nx,0:ny))
    allocate (gt(0:nx,0:ny))
    call adles(nx,ny,dx,dy,s,w,t,gw,gt)
        
	do j=0,ny
	do i=0,nx
	rw(i,j) = rw(i,j) + gw(i,j)
    rt(i,j) = rt(i,j) + gt(i,j)    		
	end do
	end do
    
    deallocate(gw,gt)
        
else if (isgs.eq.4) then !new dynamic eddy viscosity model

	allocate (gw(0:nx,0:ny))
    allocate (gt(0:nx,0:ny))
    call dyn_new(nx,ny,dx,dy,s,w,t,gw,gt)
        
	do j=0,ny
	do i=0,nx
	rw(i,j) = rw(i,j) + gw(i,j)
    rt(i,j) = rt(i,j) + gt(i,j)    		
	end do
	end do
    
    deallocate(gw,gt)

else if (isgs.eq.5) then !classical dynamic eddy viscosity model

	allocate (gw(0:nx,0:ny))
    allocate (gt(0:nx,0:ny))
    call dyn_std(nx,ny,dx,dy,s,w,t,gw,gt)
        
	do j=0,ny
	do i=0,nx
	rw(i,j) = rw(i,j) + gw(i,j)
    rt(i,j) = rt(i,j) + gt(i,j)    		
	end do
	end do
    
    deallocate(gw,gt)
            
end if


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
integer::nx,ny,NA,i,j,k,iadt
real*8::betaAD
real*8,dimension(0:nx,0:ny):: u,uf,ug,utemp

common /ADModel/ betaAD,NA,iadt


!initial guess
!k=0 
do j=0,ny
do i=0,nx
u(i,j) = uf(i,j)
end do
end do

if (iadt.eq.0) then
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

else
  
do k = 1,NA
    call filter(nx,ny,u,ug) 
		do j=0,ny
		do i=0,nx
			utemp(i,j) = uf(i,j)-ug(i,j)
        end do
        end do
    call filter(nx,ny,utemp,ug)
		do j=0,ny
		do i=0,nx
           	u(i,j) = u(i,j) + betaAD*ug(i,j)
        end do
        end do 
end do

end if

return
end

!-----------------------------------------------------------------!
!Filter
!-----------------------------------------------------------------!
subroutine filter(nx,ny,w,wf)
implicit none
integer::nx,ny,i,j
real*8 ::w(0:nx,0:ny),wf(0:nx,0:ny)
real*8, dimension (:), allocatable:: a,b
real*8, dimension(:,:),allocatable::g


allocate(g(0:nx,0:ny))

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
		call filterPade4(ny,a,b)
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
real*8 ::afil
real*8 ::u(0:n),uf(0:n)
real*8 ::alpha,beta
real*8, dimension (0:n-1):: a,b,c,r,x 

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

!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data
!Pade forth-order:  -0.5 < afil < 0.5
!Wall
!---------------------------------------------------------------------------!
subroutine filterPade4(n,u,uf)
implicit none
integer :: n,i
real*8  :: afil
real*8, dimension (0:n)	:: u,uf
real*8, dimension (1:n-1)	:: a,b,c,r,x

common /Padefilter/ afil

do i=1,n-1
a(i) = afil
b(i) = 1.0d0
c(i) = afil
end do

do i=2,n-2
r(i) = (0.625d0 + 0.75d0*afil)*u(i) &
      +(0.5d0 + afil)*0.5d0*(u(i-1)+u(i+1))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(i-2)+u(i+2))
end do

r(1) = (0.5d0 + afil)*u(1) &
      +(0.5d0 + afil)*0.5d0*(u(0)+u(2)) 

r(n-1) = (0.5d0 + afil)*u(n-1) &
      +(0.5d0 + afil)*0.5d0*(u(n)+u(n-2)) 
      

call tdma(a,b,c,r,x,1,n-1)


do i=1,n-1
uf(i)=x(i)
end do
uf(0)=u(0)
uf(n)=u(n)

return
end


!-----------------------------------------------------------------!
!Filter (trapezoidal)
!-----------------------------------------------------------------!
subroutine filtertrap(nx,ny,w,wf)
implicit none
integer::nx,ny
real*8 ::w(0:nx,0:ny),wf(0:nx,0:ny)
real*8 ::dd
integer::i,j

dd=1.0d0/16.0d0

!boundary conditions  (periodic)
wf(0,0) = 0.25d0*(w(nx-1,0)+2.0d0*w(0,0)+w(0+1,0))
wf(nx,0) = 0.25d0*(w(nx-1,0)+2.0d0*w(nx,0)+w(0+1,0))

wf(0,ny)= 0.25d0*(w(nx-1,ny)+2.0d0*w(0,ny)+w(0+1,ny))
wf(nx,ny)= 0.25d0*(w(nx-1,ny)+2.0d0*w(nx,ny)+w(0+1,ny))

do i=1,nx-1
wf(i,0) = 0.25d0*(w(i-1,0)+2.0d0*w(i,0)+w(i+1,0))
wf(i,ny)= 0.25d0*(w(i-1,ny)+2.0d0*w(i,ny)+w(i+1,ny))
end do


do j=1,ny-1  !periodic
wf(0,j) = dd*(4.0d0*w(0,j) &
             + 2.0d0*(w(0+1,j) + w(nx-1,j) + w(0,j+1) + w(0,j-1)) &
	         + w(0+1,j+1) + w(nx-1,j-1) + w(0+1,j-1) + w(nx-1,j+1))
             
wf(nx,j) = dd*(4.0d0*w(nx,j) &
             + 2.0d0*(w(0+1,j) + w(nx-1,j) + w(nx,j+1) + w(nx,j-1)) &
	         + w(0+1,j+1) + w(nx-1,j-1) + w(0+1,j-1) + w(nx-1,j+1))                         
end do



!filtering algorithm (trapezoidal filter)
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
!Routines for tridiagonal systemes
!---------------------------------------------------------------------------!

!----------------------------------------------------!
! Solution tridiagonal systems (Regular tri-diagonal)
! a:subdiagonal, b: diagonal, c:superdiagonal
! r:RHS, u:results
! S: starting index
! E: ending index
!----------------------------------------------------!
SUBROUTINE TDMS(a,b,c,r,u,S,E)
implicit none 
integer::S,E,j
real*8 ::a(S:E),b(S:E),c(S:E),r(S:E),u(S:E) 
real*8 ::bet,gam(S:E) 
  
bet=b(S)  
u(S)=r(S)/bet  
do j=S+1,E  
gam(j)=c(j-1)/bet  
bet=b(j)-a(j)*gam(j)  
u(j)=(r(j)-a(j)*u(j-1))/bet  
end do  
do j=E-1,S,-1  
u(j)=u(j)-gam(j+1)*u(j+1)  
end do  
return  
END  

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


!---------------------------------------------------------------------------!
!Routines for compact schemes
!---------------------------------------------------------------------------!

!------------------------------------------------------------------!
! c4dp:  4th-order compact scheme for first-degree derivative(up)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c4dp(u,up,h,n)
implicit none
integer :: n,i
real*8   :: h,alpha,beta
real*8 , dimension (0:n)  :: u,up
real*8 , dimension (0:n-1):: a,b,c,r,x 

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
! c4ddp: 4th-order compact scheme for second-degree derivative(upp)
!        periodic boundary conditions (0=n), h=grid spacing
!        tested
!------------------------------------------------------------------!
subroutine c4ddp(u,upp,h,n)
implicit none
integer :: n,i
real*8  :: h,alpha,beta
real*8 , dimension (0:n)  :: u,upp
real*8 , dimension (0:n-1):: a,b,c,r,x 

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

!------------------------------------------------------------------!
!c4dn:	4th-order compact scheme for the first order derivative(up)
!		neumann boundary condition u'(0)=f0, u'(n)=fn
!		3-4-3
!		tested
!------------------------------------------------------------------!
subroutine c4dn(u,up,h,n,f0,fn)
implicit none
integer :: n,i
real*8  :: h,f0,fn
real*8, dimension (0:n):: u,up
real*8, dimension (1:n-1):: a,b,c,r,x

i=1
b(i) = 1.0d0
c(i) = 5.0d0/8.0d0
r(i) = 3.0d0/2.0d0*(u(i+1)-u(i))/(h) - 1.0d0/8.0d0*f0

do i=2,n-2
a(i) = 1.0d0/4.0d0
b(i) = 1.0d0
c(i) = 1.0d0/4.0d0
r(i) = 3.0d0/2.0d0*(u(i+1)-u(i-1))/(2.0d0*h)
end do


i=n-1
a(i) = 5.0d0/8.0d0
b(i) = 1.0d0
r(i) = 3.0d0/2.0d0*(u(i-1)-u(i))/(-h) - 1.0d0/8.0d0*fn


call tdma(a,b,c,r,x,1,n-1)

!boundary conditions
up(0) = f0
do i=1,n-1
up(i) = x(i)
end do
up(n) = fn

return
end

!------------------------------------------------------------------!
!c4ddn: 4th-order compact scheme for the second order derivative(upp)
!		neumann boundary condition u'(0)=f0, u'(n)=fn
!		3-4-3
!		tested
!------------------------------------------------------------------!
subroutine c4ddn(u,upp,h,n,f0,fn)
implicit none
integer :: n,i
real*8  :: h,f0,fn
real*8, dimension (0:n):: u,upp,a,b,c,r

i=0
b(i) = 1.0d0
c(i) = 85.0d0/29.0d0
r(i) = 3.0d0/29.0d0*(-27.0d0*u(i+1)+28.0d0*u(i+2) - u(i+3))/(h*h) - 78.0d0/29.0d0*f0/h

i=1
a(i) = 127.0d0/406.0d0
b(i) = 1.0d0
c(i) =-89.0d0/406.0d0
r(i) = 12.0d0/203.0d0*(-19.0d0*u(i)+23.0d0*u(i+1)-4.0d0*u(i+2))/(h*h) - 180.0d0/203.0d0*f0/h 

do i=2,n-2
a(i) = 1.0d0/10.0d0
b(i) = 1.0d0
c(i) = 1.0d0/10.0d0
r(i) = 6.0d0/5.0d0*(u(i-1)-2.0d0*u(i)+u(i+1))/(h*h) 
end do

i=n-1
a(i) =-89.0d0/406.0d0
b(i) = 1.0d0
c(i) = 127.0d0/406.0d0
r(i) = 12.0d0/203.0d0*(-19.0d0*u(i)+23.0d0*u(i-1)-4.0d0*u(i-2))/(h*h) - 180.0d0/203.0d0*fn/h 

i=n
a(i) = 85.0d0/29.0d0
b(i) = 1.0d0
r(i) = 3.0d0/29.0d0*(-27.0d0*u(i-1)+28.0d0*u(i-2) - u(i-3))/(h*h) - 78.0d0/29.0d0*fn/h


call tdma(a,b,c,r,upp,0,n)

return
end

!----------------------------------------------------------!
!Simpson's 1/3 rule for numerical integration of f(i)
!for equally distributed mesh with interval h
!n should be power of 2
!----------------------------------------------------------!
subroutine simp1D(n,h,f,s)
implicit none
integer::n,i,nh
real*8 ::h,f(0:n),s,ds,th

nh = int(n/2)
th = 1.0d0/3.0d0*h

s = 0.0d0
do i=0,nh-1
ds = th*(f(2*i)+4.0d0*f(2*i+1)+f(2*i+2))
s = s + ds
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

!---------------------------------------------------------------------------!
!FFT-based direct elliptic solver
!using discreate fft transformation along x axis 
!solving tridiagonal matrix by thomas algorithm in y direction
!solves: d2u/dx2 + d2u/dy2 = f
!4th order Mehrstellen discretization
!top and bottom ==> homogeneous drichlet
!left and right ==> periodic
!---------------------------------------------------------------------------!
subroutine psolver(nx,ny,dx,dy,f,u)
implicit none
integer::i,j,k,nx,ny
real*8,dimension(0:nx,0:ny)::u,f
real*8::dx,dy,pi
real*8::aa,bb,cc,dd,ee,beta
real*8, dimension(:),allocatable:: data1d
real*8, dimension(:),allocatable:: a,b,c,r,x
!real*8, dimension(:),allocatable:: w0r,w0i
!real*8, dimension(:),allocatable:: wnr,wni
real*8, dimension(:,:),allocatable:: vr,vi

pi=4.0d0*datan(1.0d0)

!allocate(w0r(0:nx))
!allocate(w0i(0:nx))
!allocate(wnr(0:nx))
!allocate(wni(0:nx))

!Mehrstellen coefficients
beta = dx/dy
aa =-10.0d0*(1.0d0+beta*beta)
bb = 5.0d0 - beta*beta
cc = 5.0d0*beta*beta -1.0d0
dd = 0.5d0*(1.0d0+beta*beta)
ee = 0.5d0*dx*dx


!for lower nonhomogeneous boundary conditions
!do i=0,nx-1
	!w0r(i)=0.0d0
	!w0i(i)=0.0d0
	!do j=0,nx-1
	!w0r(i)=w0r(i)+ 1.0d0/dfloat(nx)*u(j,0)*dcos(2.0d0*pi*dfloat(i)*dfloat(j)/dfloat(nx))
	!w0i(i)=w0i(i)- 1.0d0/dfloat(nx)*u(j,0)*dsin(2.0d0*pi*dfloat(i)*dfloat(j)/dfloat(nx))
	!end do
!end do

!for upper nonhomogeneous boundary conditions
!do i=0,nx-1
	!wnr(i)=0.0d0
	!wni(i)=0.0d0
	!do j=0,nx-1
	!wnr(i)=wnr(i)+ 1.0d0/dfloat(nx)*u(j,ny)*dcos(2.0d0*pi*dfloat(i)*dfloat(j)/dfloat(nx))
	!wni(i)=wni(i)- 1.0d0/dfloat(nx)*u(j,ny)*dsin(2.0d0*pi*dfloat(i)*dfloat(j)/dfloat(nx))
	!end do
!end do

allocate(vr(0:nx,0:ny))
allocate(vi(0:nx,0:ny))
allocate(data1d(2*nx))

!finding fourier coefficients of f 
do j=0,ny
	k=1
	do i=0,nx-1
	data1d(k)  =f(i,j)
	data1d(k+1)=0.0d0
	k=k+2
	end do
	
	!invese fourier transform
	call four1(data1d,nx,-1)

	do k=1,2*nx
	data1d(k)=data1d(k)/dfloat(nx)
	end do

	k=1
	do i=0,nx-1
	vr(i,j)=data1d(k)
	vi(i,j)=data1d(k+1)
	k=k+2
	end do
end do


allocate(a(ny-1),b(ny-1),c(ny-1),r(ny-1),x(ny-1))

!solve tridiagonal system for each i
!first: real part
do i=0,nx-1

	do j=1,ny-1
	a(j) = cc + dd*2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx))     
	b(j) = aa + bb*2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx)) 
	c(j) = cc + dd*2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx)) 
	r(j) = ee*(vr(i,j)*(8.0d0+2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx))) + vr(i,j+1) + vr(i,j-1))
	end do
	!update
    !r(1)   = r(1) - a(1)*w0r(i)
	!r(ny-1)= r(ny-1) - c(ny-1)*wnr(i)

	call TDMS(a,b,c,r,x,1,ny-1)

	do j=1,ny-1
	vr(i,j)=x(j)
	end do

end do

!second: imaginary part
do i=0,nx-1

	do j=1,ny-1
	a(j) = cc + dd*2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx))     
	b(j) = aa + bb*2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx)) 
	c(j) = cc + dd*2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx)) 
	r(j) = ee*(vi(i,j)*(8.0d0+2.0d0*dcos(2.0d0*pi*dfloat(i)/dfloat(nx))) + vi(i,j+1) + vi(i,j-1))
	end do
	!update
    !r(1)= r(1) - a(1)*w0i(i)
	!r(ny-1)= r(ny-1) - c(ny-1)*wni(i)

	call TDMS(a,b,c,r,x,1,ny-1)

	do j=1,ny-1
	vi(i,j)=x(j)
	end do

end do

!finding grid values
do j=1,ny-1
  
 	k=1
	do i=0,nx-1
	data1d(k)  =vr(i,j)
	data1d(k+1)=vi(i,j)
	k=k+2
	end do

	!forward fourier transform
	call four1(data1d,nx,+1)

	k=1
	do i=0,nx-1
	u(i,j)=data1d(k)
	k=k+2
	end do
    u(nx,j)=u(0,j) !periodicity

end do


deallocate(a,b,c,r,x)
deallocate(vi,vr)
!deallocate(w0i,w0r,wni,wnr)
deallocate(data1d)

return
end



!---------------------------------------------------------------------------!
!Compute fast fourier transform for 1D data
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
!---------------------------------------------------------------------------!