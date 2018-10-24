%POD for nonlinear one dimensional PDES
%Refer Nathan Kutz POD Tutorials 1-2-3 (on YouTube)
%Romit Maulik - Oklahoma State University
clear all; close all; clc

iprob = 3;%1 - Burgers Sine Wave, 2 - Harmonic Oscillator 3 - Nonlinear Schrodinger
n_rom = 4;%Number of reduced order bases

%Burgers
if iprob==1                 %u_t + u u_x = nu u_xx
    
    nu = 0.001;L = 1; n=256;
    k = (2*pi/L)*[0:n/2-1 -n/2:-1]';%This is transposed for multiplication within ODE45 
    x2 = linspace(0,1,n+1); 
    x = x2(1:n);%Periodic boundaries n+1 not needed
    t = 0:0.01:3;
    uinit = sin(2*pi*x);%Initial condition
    
    %Initial condition for ODE45
    ut = fft(uinit);%Lets move to spectral space
    [t,utsol] = ode45('burgers_rhs',t,ut,[],k,nu);%Initial condition of ODE45 gets transposed within function - i.e. ut becomes ut'
    
    %Extracting output
    for j=1:length(t)
        usol(j,:) = ifft(utsol(j,:));
    end

    surfl(x,t,real(usol)); shading interp, colormap(hot), title('PDE Solution')

    X = usol.';
    [u,s,v]=svd(X,'econ');%SVD of our data matrix
    
elseif iprob==2                 %i u_t + 0.5 u_xx - x^2/2 u = 0
    
    L = 30; n=256;
    k = (2*pi/L)*[0:n/2-1 -n/2:-1]';%This is transposed for multiplication within ODE45 
    x2 = linspace(-L/2,L/2,n+1); 
    x = x2(1:n);%Periodic boundaries n+1 not needed
    t = 0:0.01:20;
    V = ((x.^2)./2).';%transposed for use in ODE45
      
    uinit = exp(-0.2*x.^2);
    
    %Initial condition for ODE45
    ut = fft(uinit);%Lets move to spectral space
    %Harmonic Oscillator
    [t,utsol] = ode45('harm_rhs',t,ut,[],k,V);%Initial condition of ODE45 gets transposed within function - i.e. ut becomes ut'
    
    %Extracting output
    for j=1:length(t)
        usol(j,:) = ifft(utsol(j,:));
    end

    surfl(x,t,abs(usol)); shading interp, colormap(hot), title('PDE Solution')

    X = usol.';
    [u,s,v]=svd(X,'econ');%SVD of our data matrix

elseif iprob==3                 %i u_t + 0.5 u_xx + |u|^2 u = 0
    
    L = 30; n=256;
    k = (2*pi/L)*[0:n/2-1 -n/2:-1]';%This is transposed for multiplication within ODE45 
    x2 = linspace(-L/2,L/2,n+1); 
    x = x2(1:n);%Periodic boundaries n+1 not needed
    t = linspace(0,2*pi,41);
    
    uinit = 2*sech(x);
    
    %Initial condition for ODE45
    ut = fft(uinit);%Lets move to spectral space
    %Harmonic Oscillator
    [t,utsol] = ode45('nls_rhs',t,ut,[],k);%Initial condition of ODE45 gets transposed within function - i.e. ut becomes ut'
    
    %Extracting output
    for j=1:length(t)
        usol(j,:) = ifft(utsol(j,:));
    end

    surfl(x,t,abs(usol)); shading interp, colormap(hot), title('PDE Solution')

    X = usol.';
    [u,s,v]=svd(X,'econ');%SVD of our data matrix
    
    
end

figure(2), plot(diag(s)/sum(diag(s)),'ko','Linewidth',[2]), title('Normalized Singular Values') %Plots Singular eigenvalues

figure(3), plot(real(u(:,1:3)),'Linewidth',[2]), title('POD Modes'), legend('Mode 1','Mode 2','Mode 3')%POD Modes

figure(4)
plot(real(v(:,1:3)),'Linewidth',[2]), title('V Modes (time dynamics)'), legend('Mode 1','Mode 2','Mode 3')%Time dynamics (i.e. v vector columns)

%Build a ROM

if iprob==1
    
    phi = u(:,1:n_rom);    
    
    for i=1:n_rom
       phixx(:,i) = ifft(-(k.^2).*fft(u(:,i)));%Linear operator in POD space
       phix(:,i) = ifft(i*(k).*fft(u(:,i)));%Linear operator in POD space
    end
    
    for i=1:n_rom
       a0(:,i) = u(:,i).'*uinit.';%Setting up initial condition for temporal mode
    end
    
    [t,asol] = ode45('burgers_rom_rhs',t,a0,[],phi,phix,phixx);%Galerkin projection type evolution
    
    for j=length(t)
       sum = 0;
       for i = 1:n_rom
           sum = sum + asol(j,i)*phi(:,i);
       end
       usol(j,:) = sum;                
    end
    
    figure(6)
    
    surfl(x,t,real(usol)), shading interp, colormap(hot), title('POD Solution')
    
elseif iprob==2
    
    phi = u(:,1:n_rom);   
    
    for i=1:n_rom
       phixx(:,i) = ifft(-(k.^2).*fft(u(:,i)));%Linear operator in POD space
    end
    
    for i=1:n_rom
       a0(:,i) = phi(:,i).'*uinit.';%Setting up initial condition for temporal mode
    end
    
    [t,asol] = ode45('harm_rom_rhs',t,a0,[],phi,V,phixx);%Galerkin projection type evolution
    
    for j=length(t)
       sum = 0;
       for i = 1:n_rom
           sum = sum + abs(asol(j,i))*phi(:,i);
       end
       usol(j,:) = sum;                
    end
    
    figure(6)
    surfl(x,t,abs(usol)), shading interp, colormap(hot), title('POD Solution')
    

elseif iprob==3
    
    phi = u(:,1:n_rom); 
    
    for i=1:n_rom
       phixx(:,i) = ifft(-(k.^2).*fft(u(:,i)));%Linear operator in POD space
    end
    
    for i=1:n_rom
       a0(:,i) = phi(:,i).'*uinit.';%Setting up initial condition for temporal mode
    end
    
    [t,asol] = ode45('nls_rom_rhs',t,a0,[],phi,phixx);%Galerkin projection type evolution
    
    for j=length(t)
       sum = 0;
       for i = 1:n_rom
           sum = sum + asol(j,i)*phi(:,i);
       end
       usol(j,:) = sum;                
    end
    
    figure(6)
    surfl(x,t,abs(usol)), shading interp, colormap(hot), title('POD Solution')
    
end








