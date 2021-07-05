function [A,flops] = incLU_nopiv(m,n,nb, A)
% [A,flops] = incLU_nopiv(m,n,nb, Ainput)
% perform in-core LU (nopiv) factorization
%
use_transpose_Upart = 1;
flops_Lpart = 0;
flops_Upart = 0;
flops_gemm = 0;
flops_LU = 0;



isok = (m >= n);
if (~isok),
    disp(sprintf('incLU_nopiv: m=%g n=%g ', ...
                               m,   n ));
    return;
end;

DK = zeros(nb,nb);
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

    % -----------------------------------------------------
    % C++ code may not need explicit copy for Dk, Lk, or Uk
    % -----------------------------------------------------
    Dk(1:isize,1:jsize)  = A(istart:iend, jstart:jend);
    [Lk, Uk] = lu_nopivot( Dk(1:isize,1:jsize) );

    is_square = (isize == jsize);
    if (~is_square),
      disp(sprintf('incLU_nopiv: isize=%d, jsize=%d ', ...
                                 isize,    jsize ));
      disp(sprintf('istart=%d,iend=%d,  jstart=%d, jend=%d', ...
                    istart,   iend,     jstart,    jend ));
    end;
    flops_LU = flops_LU + (2.0/3.0)*isize*isize*isize;

    % -----------------------------------------
    % over-write digonal block  with LU factors
    % -----------------------------------------
    A(istart:iend,jstart:jend) = tril(Lk(1:isize,1:jsize),-1) + ...
                                 triu(Uk(1:isize,1:jsize));

    % ----------------------------------
    % Lk * U12 = A12 => U12 = Lk \ A12
    %
    % C++ perform triangular solve as
    % TRSM( trans='NoTranspose',side='Leftside',uplo='Lower',diag='UnitDiagonal') 
    % ----------------------------------
    A( istart:iend, (jend+1):n) = Lk(1:isize,1:isize) \ A( istart:iend, (jend+1):n);
    nn = isize;
    nrhs = n - (jend+1) + 1;
    flops_Lpart = flops_Lpart + 1.0 * nn*nn*nrhs;

    % --------------------------------
    % L21 * Uk = A21 => L21 = A21/Uk
    %
    % C++ perform triangular solve as
    % TRSM( trans='NoTranspose',side='Right',uplo='Upper',diag='NonUnitDiagonal')
    % --------------------------------
    A( (iend+1):m, jstart:jend) = A( (iend+1):m, jstart:jend ) / Uk( 1:isize, 1:isize );
    nn = isize;
    nrhs = m - (iend+1) + 1;
    flops_Upart = flops_Upart + 1.0*nn*nn*nrhs;

    % ---------------------
    % GEMM update
    % A22 = A22 - L21 * U12
    % ---------------------
    i1 = (iend+1); i2 = m;
    j1 = (jend+1); j2 = n;
    k1 = jstart; k2 = jend;

    isize = (i2-i1+1);
    jsize = (j2-j1+1);
    ksize = (k2-k1+1);

%   -------------------------------
%   may use fp16 in Lpart and Upart
%   or use transpose storage
%   -------------------------------
    Lpart(1:isize,1:ksize) = A(i1:i2,k1:k2);

    if (use_transpose_Upart),
      Upart(1:jsize,1:ksize) = transpose(A(k1:k2,j1:j2));
    else 
      Upart(1:ksize,1:jsize) = A(k1:k2,j1:j2);
    end;


%     ------------------------------------------
%     equivalent to 
%     A( i1:i2, j1:j2) = A(i1:i2, j1:j2) - ...
%           A( i1:i2, k1:k2) * A( k1:k2, j1:j2);
%     ------------------------------------------

    if (use_transpose_Upart),
      A( i1:i2, j1:j2) = A(i1:i2, j1:j2) - ...
          Lpart( 1:isize, 1:ksize) * transpose(Upart( 1:jsize,1:ksize ));
    else
      A( i1:i2, j1:j2) = A(i1:i2, j1:j2) - ...
          Lpart( 1:isize, 1:ksize) * Upart( 1:ksize, 1:jsize );
    end;

    mm = isize;
    nn = jsize;
    kk = ksize;
    flops_gemm = flops_gemm + 2.0*mm*nn*kk;


flops.flops_gemm = flops_gemm;
flops.flops_Lpart = flops_Lpart;
flops.flops_Upart = flops_Upart;
flops.flops_LU = flops_LU;

end;
