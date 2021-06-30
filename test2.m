m = 4;
A = ones(m,m);
for i=1:m, A(i,i) = 2; end;
nb  = 1;
width = 1;
L = oochol(m, A, nb, width );
