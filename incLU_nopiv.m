function [A] = incLU_nopiv(m,n,nb, Ainput)
% [A] = incLU_nopiv(m,n,nb, Ainput)
% perform in-core LU (nopiv) factorization
%
A = Ainput;

isok = (m >= n);
if (~isok),
    disp(sprintf('incLU_nopiv: m=%g n=%g ', ...
                               m,   n ));
    return;
end;

for jstart=1:nb:n,
    jend = min(n,jstart+nb-1);
    jsize = (jend-jstart+1);

    istart = jstart;
    iend = min(m,jend);
    isize = (iend-istart+1);
    % ---------------------
    % factor diagonal block
    % ---------------------

    isok = (isize == jsize);
    if (~isok),
        disp(sprintf('incLU_nopiv: isize=%g, jsize=%g ', ...
                                   isize,    jsize ));
        
        disp(sprintf('incLU_nopiv: istart=%g,iend=%g,jstart=%g,jend=%g',...
                                   istart,iend,jstart,jend));
        return;
    end;

    Dk = A(istart:iend, jstart:jend);
    [Lk, Uk] = lu_nopivot( Dk );

    % ----------------------------------
    % L11 * U12 = A12 => U12 = L11 \ A12
    % ----------------------------------
    A( istart:iend, (jend+1):n) = Lk(1:isize,1:isize) \ A( istart:iend, (jend+1):n);
    % --------------------------------
    % L21 * U11 = A21 => L21 = A21/U11
    % --------------------------------
    A( (iend+1):m, jstart:jend) = A( (iend+1):m, jstart:jend ) / Uk( 1:isize, 1:isize );

    % ---------------------
    % GEMM update
    % A22 = A22 - L21 * U12
    % ---------------------
    i1 = (iend+1); i2 = m;
    j1 = (jend+1); j2 = n;
    k1 = jstart; k2 = jend;

    A( i1:i2, j1:j2) = A(i1:i2, j1:j2) - ...
          A( i1:i2, k1:k2) * A( k1:k2, j1:j2);


    % -----------------------------------------
    % over-write digonal block  with LU factors
    % -----------------------------------------
    A(istart:iend,jstart:jend) = tril(Lk,-1) + triu(Uk);

end;
