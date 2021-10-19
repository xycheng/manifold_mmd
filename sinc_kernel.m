function y = sinc_kernel(x)


y=sin(pi/2*x)./(pi/2*x);
y( abs(x) < 1e-10) =1;


return;