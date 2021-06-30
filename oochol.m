function L = oochol( m, Ain, nb, width )
% L = oochol( m, Ain, nb, width )
%
% out of core cholesky factorization
% nb is block size, width is width of y panel
% note width should be a multiple of nb
t1 = cputime;
L = chol(Ain(1:m,1:m))';
t2 = cputime;
disp(sprintf('chol took %g ', t2-t1));

isok = (mod(width,nb) == 0);
if (~isok),
    sprintf(sprintf('oochol: nb=%g, width=%g', ...
                             nb,    width ));
end;
% -------------------------
% convert to block storage
% -------------------------
nblk = (m + nb-1)/nb;
A = zeros( nb,nb,nblk,nblk );
for jb=1:nblk,
for ib=1:nblk,
     A(:,:,ib,jb) = eye(nb,nb);
end;
end;
for jb=1:nblk,
for ib=jb:nblk,
   ia = 1 + (ib-1)*nb;
   ja = 1 + (jb-1)*nb;
   iaend = min(m, ia+nb-1);
   jaend = min(m, ja+nb-1);
   iasize = iaend-ia+1;
   jasize = jaend-ja+1;
   A(1:iasize,1:jasize,ib,jb) = Ain( ia:iaend, ja:jaend );
end;
end;

% ----------------------
% allocate X and Y panel
% ----------------------
X = zeros( nb,nb, nblk );
Y = zeros( nb,nb, nblk*width );

for jbstart=1:width:nblk,
   jbend = min( nblk, jbstart + width-1);
   jbsize = jbend - jbstart + 1;

   % -----------------
   % copy A to Y panel
   % -----------------
   mblk = nblk - jbstart + 1;
   ibstart = jbstart;
   for jb=jbstart:jbend,
     for ib=jb:nblk,
         ii = 1 + (ib-ibstart);
         jj = 1 + (jb-jbstart);
         ip = ii + (jj-1)*mblk;
         Y(:,:,ip) = A(:,:,ib,jb);
     end;
   end;

   % -----------------------------
   % perform updates with X panel
   % -----------------------------
   for jb=1:(jbstart-1),
     % -----------------
     % copy into X panel
     % -----------------
     for ib=jbstart:nblk,
          X(:,:,ib) = A(:,:,ib,jb);
     end;
     
     % -----------------
     % symmetric update
     % -----------------
     for jb=jbstart:jbend,
        ii = (jb-jbstart) + 1;
        jj = ii;
        iy = ii + (jj-1)*mblk;
        ix = jb;
        Y(:,:,iy) = Y(:,:,iy) - X(:,:,ix) * X(:,:,ix)';
     end;
     % ------------------
     % offdiagonal update
     % ------------------
     for jb=jbstart:jbend,
       for ib=(jb+1):nblk,
         ii = (ib-jbstart) + 1;
         jj = (jb-jbstart) + 1;
         iy = ii + (jj-1)*mblk;

         Y(:,:,iy) = Y(:,:,iy) - X(:,:,ib) * X(:,:,jb)';
        end;
      end;
    end;


    % --------------------
    % incore factorization of Y
    % --------------------

    for jb=1:jbsize,
       % ---------------------
       % factor diagonal block
       % ---------------------
       jj = jb;
       ii = jj;
       idiag = ii + (jj-1)*mblk;
       Y(:,:,idiag) = chol( Y(:,:,idiag) )';

        
       % ------------------
       % update column of L
       % Lij * Lii' = Aij
       % Lij = Aij/Lii'
       % ------------------
       for ib=(jb+1):mblk,
         ii = ib;
         jj = jb;
         ky = ii + (jj-1)*mblk;

         Y(:,:,ky) = Y(:,:,ky)/Y(:,:,idiag);
       end;

       % ----------------
       % symmetric update
       % ----------------
       for jb2=(jb+1):jbsize,
         jj = jb2;
         ii = jj;
         kc = ii + (jj-1)*mblk;

         jj = jb;
         ii = jb2;
         ka = ii + (jj-1)*mblk;
         Y(:,:,ic) = Y(:,:,ic) - Y(:,:,ia)*Y(:,:,ia)';
       end;

       % ------------------
       % offdiagonal update
       % ------------------
       for jb2=(jb+1):jbsize,
         for ib2=(jb2+1):mblk,
           jc = jb2;
           ic = ib2;
           kc = ic + (jc-1)*mblk;

           ja = jb;
           ia = ic;
           ka = ia + (ja-1)*mblk;

           jj = jb;
           ii = jb2;
           kb = ii + (jj-1)*mblk;
           Y(:,:,kc) = Y(:,:,kc) - Y(:,:,ka) * Y(:,:,kb)';
         end;
       end;
     
    end;

    % -----------
    % copy Y to A
    % -----------
    ibstart = jbstart;
    for jb=jbstart:jbend,
      for ib=jbstart:nblk,
         ii = 1+(ib-ibstart);
         jj = 1+(jb-jbstart);
         ip = ii + (jj-1)*mblk;
         A(:,:,ib,jb) = Y(:,:,ip );
      end;
    end;
end;

 % check error
 maxerr = 0;
 for jb=1:nblk,
 for ib=jb:nblk,
   for j=1:nb,
   for i=j:nb,
     ia = i + (ib-1)*nb;
     jb = j + (jb-1)*nb;
     err = 0;
     ia_ok = (1 <= ia) & (ia <= m);
     ja_ok = (1 <= ja) & (ja <= m);
     is_lower = (ia >= ja);
     if (ia_ok & ja_ok & is_lower),
       lij = A(i,j,ib,jb);
       err = abs( L(ia,ja) - lij );
     end;
     maxerr = max( maxerr, err );
    end;
    end;
  end;
  end;

  disp(sprintf('maxerr %g ', maxerr ));

     

















  


