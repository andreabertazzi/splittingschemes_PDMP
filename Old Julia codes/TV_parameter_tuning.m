addpath('images')
addpath('functions')
%addpath('toolbox_signal') %This comes from Gabriel Peyre numerical tours website https://github.com/gpeyre/numerical-tours
%addpath('toolbox_general') %This comes from Gabriel Peyre numerical tours website https://github.com/gpeyre/numerical-tours

%% Load images
    blur_length=25;
    bsnr=30;
    name='cman';
    %name='handwritten';
    dirname = 'results/deblur_TV';
    true_image = load_image(name); %loads the image
    true_image=true_image/255;
    %n=64;
    %true_image=true_image(floor(n/4):floor(5*n/4),floor(1.5*n):floor(2.5*n));  %For cropped Cameraman image
%% Optimising theta
        x=true_image;
        sigma=1/255;
        sigma2=sigma^2;
        dimX = numel(x);
        %%%%%  blur operator
            [A,AT,H_FFT,HC_FFT]=uniform_blur(length(x),blur_length);
            evMax=max_eigenval(A,AT,size(x),1e-4,1e4,0);%Maximum eigenvalue of operator A.

             y = A(x) + sigma*randn(size(A(x)));


    %%%% Parameter Setup
    op.warmup = 300 ;% number of warm-up iterations with fixed theta for MYULA sampler
    op.lambdaMax = 2; % max smoothing parameter for MYULA
    lambdaFun= @(Lf) min((5/Lf),op.lambdaMax);
    gamma_max = @(Lf,lambda) 1/(Lf+(1/lambda));%Lf is the lipschitz constant of the gradient of the smooth part
    %op.gammaFrac=0.98; % we set PULA.step_size=op.PULA.step_sizeFrac*PULA.step_size_max
    op.gammaFrac=0.9; % we set PULA.step_size=op.PULA.step_sizeFrac*PULA.step_size_max

    op.samples =1500; % max iterations for SAPG algorithm to estimate theta
    op.stopTol=1e-3; % tolerance in relative change of theta_EB to stop the algorithm
    op.burnIn=20;	% iterations we ignore before taking the average over iterates theta_n

    op.th_init = 0.01 ; % theta_0 initialisation of the SAPG algorithm
    op.min_th=1e-3; % projection interval Theta (min theta)
    op.max_th=20; % projection interval Theta (max theta)

        %%% Experiment setup: functions related to Bayesian model
        %%%% Regulariser
        % TV norm
        op.g = @(x) TVnorm(x); %g(x) for TV reg
        chambolleit = 25;
        % Proximal operator of g(x)         
        op.proxG = @(x,lambda,theta) chambolle_prox_TV_stop(x,'lambda',lambda*theta,'maxiter',chambolleit);
        Psi = @(x,th) op.proxG(x,1,th); % define this format for SALSA solver
                         
        %%%% Likelihood (data fidelity)
        op.f = @(x) (norm(y-A(x),'fro')^2)/(2*sigma2); % p(y|x)∝ exp{-op.f(x)}
        op.gradF = @(x) real(AT(A(x)-y)/sigma2);  % Gradient of smooth part f
        Lf = (evMax/sigma)^2; % define Lipschitz constant of gradient of smooth part

    
        % delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
    op.d_exp =  0.8;
    op.d_scale =  0.1/op.th_init;


        % We use this scalar summary to monitor convergence
        op.logPi = @(x,theta) -op.f(x) -theta*op.g(x);
        
        %%%% Set algorithm parameters that depend on Lf
        op.lambda=lambdaFun(Lf);%smoothing parameter for MYULA sampler
        op.gamma=op.gammaFrac*gamma_max(Lf,op.lambda);%discretisation step MYULA
        
        
        %%% Run SAPG Algorithm 1 to compute theta_EB
        [theta_EB,results]=SAPG_algorithm_1(y,op);
       
theta = results.last_theta;

display(['We have tuned theta =',num2str(theta)])