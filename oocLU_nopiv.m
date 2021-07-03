function [A] = oocLU_nopiv( n, Ainput, nb, incore_size )
% [A] = oocLU_nopiv( n, Ainput, nb, memsize )
% perform out of out of core LU (nopivot) factorization
% using approximately incore_size amount of in-core memory
%

idebug = 1;
use_transpose_Upart = 1;

% ---------------------
% want incore memory size to be multiple of nb width column panels
% ---------------------
incore_blocks = floor( incore_size/(n*nb) ); 

% ------------
% set width of X-panel to get sufficient performance in I/O and GEMM
% the rest of memory dedicated to Y-panel
% ------------
x_blocks = 1;
y_blocks = incore_blocks - x_blocks;

% ----------------------
% C++ code will over-write original matrix
% copy only for matlab, no need to make a copy
% in C++ code
% ----------------------
A = zeros(n,n);
A(1:n,1:n) = Ainput(1:n,1:n);


isok = (y_blocks >= x_blocks);
if (~isok),
    disp(sprintf('oocLU_nopiv: x_blocks %g, y_blocks %g ', ...
                               x_blocks,    y_blocks ));
    return;
end;


% ---------------------
% setup in-core matrix
% ---------------------
x_width = (x_blocks*nb);
y_width = (y_blocks*nb);
if (idebug >= 1),
  disp(sprintf('n=%d, incore_size=%d, nb=%d, x_width=%d, y_width=%d', ...
                n,    incore_size,    nb,    x_width,    y_width));
end;
Y = zeros( n, y_width);
X = zeros( n, x_width);

for jstarty=1:y_width:n,
    jendy = min(n, jstarty + y_width-1);
    jsizey = (jendy - jstarty + 1);

    % ------------------------
    % copy into in-core Y-panel matrix
    % ------------------------
    Y(1:n,1:jsizey) = A(1:n, jstarty:jendy );

    % ----------------------------------------------
    % perform update from previously compute factors
    % ----------------------------------------------
    for jstartx=1:x_width:(jstarty-1),
        jendx = min( (jstarty-1), jstartx + x_width-1);
        jsizex = (jendx - jstartx + 1);

        % ----------------------------------
        % copy into in-core  X-panel matrix
        % copy whole panel for simplicity and
        % a single contigous bulk transfer
        % ----------------------------------
        istartx = jstartx;
        iendx = jendx;
        isizex = (iendx - istartx + 1);
        X(:,1:jsizex) = A(:, jstartx:jendx );

        % -------------------------
        % diagonal block in X panel
        % -------------------------

        if (idebug >= 2),
         disp(sprintf('jstartx=%d, jendx=%d', ...
                       jstartx,    jendx ));

         disp(sprintf('isizex=%d, jsizex=%d', ...
                       isizex,    jsizex ));
        end;

        Dk = zeros(isizex,jsizex);
        Dk(1:isizex,1:jsizex) = X( istartx:iendx, 1:jsizex );

        Lk = tril(Dk,-1) + eye(isizex,jsizex);
        Uk = triu( Dk );

        % ---------------
        % L11 * Upart = A12
        % or Upart = L11\A12
        % ---------------
        Y( istartx:iendx, 1:jsizey) = Lk(1:isizex,1:jsizex)\Y( istartx:iendx, 1:jsizey);

        % ---------------------------------------------
        % may need to copy to fp16 or transpose storage
        % ---------------------------------------------
        if (use_transpose_Upart),
          Upart = zeros( jsizey,jsizex );
          Upart( 1:jsizey, 1:jsizex) = transpose( Y(istartx:iendx,1:jsizey) );
        else
          Upart = zeros(jsizex,jsizey);
          Upart( 1:jsizex,1:jsizey) = Y( istartx:iendx, 1:jsizey);
        end;


        % -----------
        % GEMM update
        % may need to copy to fp16 or transpose storage 
        % in Lpart or Upart
        % -----------
        i1 = (iendx+1);
        i2 = n;
        isize = (i2-i1+1);

        Lpart(1:isize,1:jsizex) = X( i1:i2, 1:jsizex);
        if (use_transpose_Upart),
          Y( i1:i2, 1:jsizey) = Y(i1:i2, 1:jsizey) - ...
                 Lpart( 1:isize, 1:jsizex) * transpose(Upart( 1:jsizey, 1:jsizex));
        else
          Y( i1:i2, 1:jsizey) = Y(i1:i2, 1:jsizey) - ...
                 Lpart( 1:isize, 1:jsizex) * Upart( 1:jsizex, 1:jsizey);
        end;

    end;

    % -------------------------------
    % previous updates performed
    % ready for in-core factorization of Y-panel
    % -------------------------------
    i1 = jstarty;
    i2 = n;
    mm = i2-i1+1;
    nn = jsizey;
    if (idebug >= 2),
        disp(sprintf('jstarty=%d, n=%d, i1=%d, i2=%d', ...
                      jstartx,    n,    i1,    i2 ));
        disp(sprintf('mm=%d, nn=%d ', ...
                      mm,    nn));
    end;
    Y(i1:i2, 1:nn) = incLU_nopiv( mm,nn,nb, Y(i1:i2, 1:nn) );

    % --------------------------------------
    % copy result back to out-of-core matrix
    % --------------------------------------
    A(:, jstarty:jendy) = Y( :, 1:jsizey );
end;


