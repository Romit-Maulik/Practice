function[p,P]=DEIMGappy(U)
[n,m]=size(U);
fac = 3;
p=zeros(fac,m);
NodesUnused = 1:n;
[~,idx]=sort(abs(U(:,1)),'descend');
p(:,1) = idx(1:fac);
I = eye(n,n);
U_temp = U;
I(p(:,1),1) = 1;
P = zeros(n,fac*m);
P(:,1:fac) = I(:,p(:,1));
NodesUnused = setdiff(NodesUnused, p(:,1));
for ii=2:m
    c=pinv((P(:,1:(ii-1)*fac)'*U_temp(:,1:ii-1)))*(P(:,1:(ii-1)*fac)'*U(:,ii));
    r=U(:,ii)-U_temp(:,1:(ii-1))*c;
    [~,idx]=sort(abs(r),'descend');
    membership =ismember(idx,NodesUnused);
    count = 1;
    iter = 1;
    while(count <= fac && iter <=length(membership))
        if(membership(iter) == 1)
            p(count,ii) = idx(iter);
            count = count + 1;
            NodesUnused = setdiff(NodesUnused, idx(iter));
            iter = iter + 1;
        else
            iter = iter + 1;
        end
            
    end
    P(:,(ii-1)*fac+1:ii*fac) = I(:,p(:,ii));
    if(numel(unique(p(:,1:ii))) ~= fac*ii)
        keyboard
    end
end

end