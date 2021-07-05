      idebug = 0;

      n = 1013;
      A = 2*rand(n,n)-1;
      incore_size = floor( n*n * 0.3 );
      nb = 9;
      [ALU] = oocLU_nopiv( n, A, nb, incore_size );
      L = tril(ALU,-1) + eye(size(ALU));
      U = triu(ALU);
      R = A - L * U;
      err = norm( R , 1 );
      disp(sprintf('err = %g, norm(A) = %g ', err, norm(A,1) ));

      if (idebug >= 1),
        [Lk, Uk] = lu_nopivot( A );
        disp(sprintf('norm(Lk-L,1)=%g, norm(Uk-U,1)=%g', ...
                      norm(Lk-L,1),    norm(Uk-U,1) ));
        disp(sprintf('norm(Lk,1)=%g, norm(Uk,1)=%g', ...
                      norm(Lk,1),    norm(Uk,1) ));

      end;
