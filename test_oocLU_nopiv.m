      idebug = 0;

      n = 101;
      A = 2*rand(n,n)-1;
      incore_size = floor( n*n * 0.3 );
      nb = 9;
      [ALU] = oocLU_nopiv( n, A, nb, incore_size );
      L = tril(ALU,-1) + eye(size(ALU));
      U = triu(ALU);
      R = A - L * U;
      err = norm( R , 1 );
      disp(sprintf('err = %g ', err));

      if (idebug >= 1),
        [Lk, Uk] = lu_nopivot( A );
        subplot(2,1,1); spy( abs(Lk-L) > 1e-5 );
        subplot(2,1,2); spy( abs(Uk-U) > 1e-5 );
      end;
