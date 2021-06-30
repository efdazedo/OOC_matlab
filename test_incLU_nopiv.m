m = 100;
n = 47;
nb = 8;
Aorg = rand(m,n);
ALU = incLU_nopiv( m,n,nb, Aorg );

L = tril( ALU,-1) + eye( size(ALU) );
U = triu( ALU );

disp(sprintf('size(L)=%g %g ', ...
    size(L,1), size(L,2) ));

disp(sprintf('size(U)=%g %g ', ...
    size(U,1), size(U,2) ));

minmn = min(m,n);
err = norm( Aorg - L(1:m,1:minmn) * U(1:minmn,1:n), 1 );
disp(sprintf('m=%g, n=%g, nb=%g err=%g', ...
              m,    n,    nb,   err ));


