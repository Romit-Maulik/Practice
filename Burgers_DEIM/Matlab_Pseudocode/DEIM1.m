function[p,P]=DEIM1(U)
[n,m]=size(U);
p=zeros(1,m);
CC = abs(U(:,1));
[~,p(1,1)]=max(abs(U(:,1)));
U_temp = U(:,1);
I = eye(n,n);
P = I(:,p(1,1));
for ii=2:m
    c=(P'*U_temp)\(P'*U(:,ii));
    %keyboard
    r=U(:,ii)-U_temp*c;
    [~,p(1,ii)]=max(abs(r));
    U_temp=[U_temp, U(:,ii)];
    P=[P I(:,p(1,ii))];
end

end