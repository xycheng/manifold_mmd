% this file reproduces the result in figure 2 of the paper
%
%   X. Cheng and Y. Xie. "Kernel MMD two-sample tests for manifold data".
%
%   ArXiv link: https://arxiv.org/abs/2105.03425
%

clear all; rng(2021);

use_noisy_data = 1;    % change to 0 to obtain results on clean images

%% load image data 

tmp = load('img_8.mat'); 
img= tmp.img;
img = reshape(img,28,28);

%% visulize of density on S1
delta = 0.5;
shift_t = 0.2;

% visualize of distribution of shifts
p0_func = @(t,  ome)  sin( ome * 2*pi *t);
Fp0_func = @(t,  ome)  (1- cos(ome * 2*pi * t))/(ome*2*pi);

q0_func = @(t,  ome)  sin( ome * 2*pi *( t-shift_t) );
Fq0_func = @(t,  ome)  (cos(ome * 2*pi * (- shift_t)) ...
    - cos(ome * 2*pi * (t - shift_t)))/(ome*2*pi);

p_func = @(t)  (3   +p0_func(t,  2)+ 0.6*p0_func(t, 5))/3;
Fp_func = @(t) (3*t+Fp0_func(t,  2)+ 0.6*Fp0_func(t, 5))/3;

q_func = @(t)  ((3+delta) + q0_func(t,  2))/(3+delta);
Fq_func = @(t) ((3+delta)*t+ Fq0_func(t,  2))/(3+delta);


% grid used to sample ~ pdt
dtgrid = 1e-5;
tgrid = (0: dtgrid : 1)';
Fpgrid = Fp_func(tgrid);
Fqgrid = Fq_func(tgrid);


%
dtvis = 1e-3;
tvis = (0: dtvis  : 1)';

figure(1),clf;
subplot(121), hold on;
plot( tvis, p_func(tvis),'.-');
plot( tvis, q_func(tvis),'.-');
grid on;
legend('p', 'q')
set(gca,'FontSize',20)
subplot(122), hold on;
plot( tvis, Fp_func(tvis),'.-');
plot( tvis, Fq_func(tvis),'.-');
grid on;
legend('Fp', 'Fq', 'Location','southeast')
set(gca,'FontSize',20)


%%

%% two-sample testing

n = 200;
nX= n;
nY= n;

sig_scale_list= [1/4, 1/2, 1, 2, 4 ];
nrow=numel(sig_scale_list);

w_list = [10:5:40]';
    % set to 
    %   w_list = [10:2:40]';
    % to reproduce the restul in the paper

dim_list = w_list.^2;


ncol = numel(dim_list);

%level of test
alp=.05; 

% Monte-carlo runs
nrun = 50; 
    %nrun = 50 takes about 5 min to run (on a 15-inch macbook pro,
    %processor 2.9 GHz Quad-Core Intel Core i7)
    %set to 500 to reproduce figure in the paper, takes longer time to run

%number of permutations to estimate threshold
numperm= 400; 

%%
vote_g = zeros(nrow, ncol, nrun);

vote_g_mid = zeros(ncol, nrun);
sig_median_all =zeros(ncol, nrun);

vote_g_t = zeros(nrow, ncol, nrun);


tic,
for icol = 1:ncol
    size_w = w_list(icol);
    dim = size_w^2;
    
    
    fprintf('dim=%d:',dim)
    
    for irun = 1:nrun
        
        if mod(irun,10)==0
            fprintf('-%d-',irun)
        end
        
        % sample the angles
        tX = sample_by_cdf_1d( tgrid, Fpgrid, nX);
        tY = sample_by_cdf_1d( tgrid, Fqgrid, nY);
        
        max_angle = 90; 
        
        X = angle_to_rotate_img( img, size_w, tX, max_angle);
        Y = angle_to_rotate_img( img, size_w, tY, max_angle);
        
        %% add noise
        sig0_X = 20; 
        
        if use_noisy_data 
            noise_sig = sig0_X; %image is of [0,255]
            
            X = X+ randn(size(X))*noise_sig;
            Y = Y+ randn(size(Y))*noise_sig;
        end
        
        %% gaussian mmd to the high dimensional data
        D2=euclidean_dis2(X,Y);
        
        for irow=1:nrow
            
            sig = sqrt(dim)*sig0_X*sig_scale_list(irow);
            
            K1=exp(-D2/(2*sig^2));
            
            eta1=calculate_kernel_mmd2(K1,nX,nY);
            
            etastore=zeros(1,numperm);
            for iboot=1:numperm
                idx=randperm(nX+nY);
                etastore(iboot)=calculate_kernel_mmd2(K1(idx,idx),nX,nY);
            end
            talp=quantile(etastore,1-alp);
            
            vote_g(irow,icol,irun)=double(eta1>talp);
        end
        
        %% gaussian mmd with median sig
        dis = squareform( sqrt(D2));
        sig = median(dis);
        sig_median_all(icol, irun) = sig;
        
        K1=exp(-D2/(2*sig^2));
        
        eta1=calculate_kernel_mmd2(K1,nX,nY);
        
        etastore=zeros(1,numperm);
        for iboot=1:numperm
            idx=randperm(nX+nY);
            etastore(iboot)=calculate_kernel_mmd2(K1(idx,idx),nX,nY);
        end
        talp=quantile(etastore,1-alp);
        
        vote_g_mid(icol,irun)=double(eta1>talp);
        
    end
    fprintf('\n')

end

toc

%% compute power

powg=zeros(nrow,ncol);

for icol=1:ncol
    tmp=reshape(vote_g(:,icol,:),nrow,nrun);
    powg(:,icol)=sum( tmp,2)/nrun;
end

disp('-- Gmmd on X Y --')
disp(powg*100)

%
powg_mid=zeros(1,ncol);

for icol=1:ncol
    tmp=reshape(vote_g_mid(icol,:),1,nrun);
    powg_mid(icol)=sum( tmp,2)/nrun;
end

disp('-- Gmmd on X Y, median sig --')
disp(powg_mid*100)

%
figure(12),clf; 
hold on;
for irow =1:nrow
    plot(dim_list,powg(irow,:),'x-','LineWidth',2,...
       'DisplayName', [num2str(sig_scale_list(irow)),'\sigma_0']);
end

plot(dim_list, powg_mid,'x--','DisplayName','median' )
grid on;
if use_noisy_data
    title(sprintf('testing power, $n_X=n_Y=%d$, noisy data',n),'Interpreter','latex')
else
    title(sprintf('testing power, $n_X=n_Y=%d$, clean data',n),'Interpreter','latex')
end
xlabel('ambient dimension $m$','Interpreter','latex')
set(gca,'FontSize',20);
axis([100, 1600, 0.3, 1])

%% visualize one image
id =3;

figure(11),clf;
imagesc( reshape( X(id,:), [size_w, size_w]))
colormap(1-gray); axis off;

%%
return;

