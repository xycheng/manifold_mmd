function eta=calculate_kernel_mmd2(K,nX,nY)
assert(size(K,1)==nX+nY);
KXX=K(1:nX,1:nX);
KXY=K(1:nX,nX+1:nX+nY);
KYY=K(nX+1:nX+nY,nX+1:nX+nY);
eta=mean(KXX(:))+ mean(KYY(:))-2*mean(KXY(:));
end