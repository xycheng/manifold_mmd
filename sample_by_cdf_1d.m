function tt = sample_by_cdf_1d( tgrid, Fgrid, n)

ff = sort( rand( n,1), 'ascend'); %unif(0,1)
tt = interp1(Fgrid,tgrid,ff ,'spline');

return;
