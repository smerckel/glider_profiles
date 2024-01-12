% calculates filter coefficients for 2nd order filter

Hs:(c1+c2*s+c3*s^2)/(d1+d2*s+d3*s^2);

Hz:factor(subst(s=-2/T*(1-1/z)/(1+1/z),Hs));
Hz2:ratsimp(Hz,z);
Hz_num:num(Hz2);
Hz_denom:denom(Hz2);

a0:subst(z=0,Hz_num);
tmp:Hz_num-a0;
a1:subst(z^2=0,tmp);
a2:Hz_num-a1-a0;

b0:subst(z=0,Hz_denom);
tmp:Hz_denom-b0;
b1:subst(z^2=0,tmp);
b2:Hz_denom-b1-b0;

a1:subst(z=1,a1);
a2:subst(z=1,a2);
b1:subst(z=1,b1);
b2:subst(z=1,b2);

a0s:a0/b0;
a1s:a1/b0;
a2s:a2/b0;
%b0==1
b1s:b1/b0;
b2s:b2/b0;
