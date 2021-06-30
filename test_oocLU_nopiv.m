      n = 77;
      A = rand(n,n);
      incore_size = (n*n)/2;
      nb = 4;
      [ALU] = oocLU_nopiv( n, A, nb, incore_size );
