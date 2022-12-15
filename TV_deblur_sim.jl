using Plots
using Images
using JLD2  # For saving data
using FFTW  # For fast fourier transform
using Distributions  # For probability distributions
using Random
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("imaging_utils.jl")

run_ZZS = false
run_BPS = false
run_ULA = false
run_ZZS_cts = true
# factor_ULA = 1.98
# factor_ZZS = 1000
factor_ULA = 1.98
factor_ZZS = 3000
refresh_rate = 1e-2;

blur_length = 9;
N=10^3;   #Number of steps per sample we store
num_thin = 1 * 10^3;    # Number of samples we store
num_thin_ctszzs = 1 * 10^5
thin_low = 3
BSNRdb = 40;
# BSNRdb = false

# name = "cman";
name = "handwritten";
# name = "cmansmall";
f0 = Float64.(load(string("Images/",name,".png")));

# if maximum(maximum(f0))>1
#     f0=f0/255
# end
println(string("The maximum pixel is ", maximum(maximum(f0))))

#f0=f0[1:20,1:20]
p_true=plot(Gray.(f0),title="Ground Truth",axis=([], false))

f0_vec=vec(reshape(f0,:,1))
(nx,ny)=size(f0);

## Create Observation

h=ones(blur_length);
lh=length(h);
h = h/sum(h);
h=vcat(h, zeros(nx-lh));
h=cshift(h,-(lh-1)/2);
h = reshape(h,:,1)*reshape(h,1,:);
hF =fft(h);

# display(plot(sort(vec(abs.(hF)^2)), seriestype=:scatter, title="Eigenvalues of Blur operator"))
hF_vec=vec(reshape(h,:,1));
hFC=hF';

y=real(ifft(fft(f0).*hF));
y_vec=vec(reshape(y,:,1))


# sigma=1/255;
# #BSNR=20;
# dimX=nx*ny;
# #sigma = norm(y-mean(mean(y)))/sqrt(dimX*10^(BSNRdb/10));
# sigma2=sigma*sigma;
# #y=y+sigma*randn(nx,ny);
if BSNRdb == false
    sigma=1/255
else
    dimX=nx*ny;
    sigma = norm(y.-mean(mean(y)))/sqrt(dimX*10^(BSNRdb/10));
    other_sigma = 1/255
    println("We have set sigma to be ", sigma, " instead of $other_sigma")
end
sigma2=sigma*sigma;
Random.seed!(1234)
y=y+sigma*randn(nx,ny);
p_noise=plot(Gray.(y), title="Observation",axis=([], false))
Random.seed!()


respath=string("results/deblur_TV/",name)
if !isdir(respath)
    mkdir(respath)
end

y_vec=vec(reshape(y,:,1))

yF=fft(y);
yF_vec=vec(reshape(yF,:,1));

## Likelihood

A=x->reshape(real(ifft(fft(reshape(x,size(f0))).*hF)),:,1);
AT=x->reshape(real(ifft(fft(reshape(x,size(f0))).*hFC)),:,1);

# fLike=x->norm(A(x)-y)^2/(2*sigma2);
yFf=vec(reshape(yF,:,1));
y_vec=vec(reshape(y,:,1));
println("The PSNR between the true image and the observation is ",psnr(y_vec,f0_vec))
gradf = x-> AT(A(x)-y_vec)/sigma2

##prior
# g=x->s/sigma2*norm(x)^2;
#gradg = x-> s_max*x/sigma2
if name=="cman"
    alpha=6.5765; #This is for the full cameraman image
elseif name=="cmansmall"
    alpha=5.3966;
elseif name=="handwritten"
    alpha= 5.143;    #This is for the handwritten digit
else
    print("The parameter alpha has not been tuned, please use MATLAB algorithm SAPG to tune alpha")
    alpha=0.044;
end

lambda_prox=5*sigma2;
#@pyimport chambolle_prox_TV.py

#proxg=x->pycall(chambolle_prox_TV.chambolle_prox_TV, PyAny, reshape(x,size(f0)),alpha*lambda_prox,20);
include("chambolle_prox.jl")
#pyimport(portal)
#proxg =x ->pycall(portal.algorithms.chambollepock.chambolle_pock_tv, x, )
proxg = x -> chambolle_prox_TV(reshape(x,size(f0)), alpha*lambda_prox,20)

#Stability barrier
#For ULA the stability barrier is 2/L where L is the Lipschitz constant
L=1/sigma2+alpha/lambda_prox;
delta_max=1/L; #Here we are using half the stability barrier to be sure we are clear of it


gradU=x->vec(gradf(x)+(x-reshape(proxg(x),:,1))/lambda_prox)
x_init=copy(y_vec);
nxi=length(x_init)


if run_ZZS_cts

    v_init = rand(Bernoulli(0.5),size(x_init));
    v_init=Int.(v_init);
    v_init=2*v_init .-1;
    ZZScts_sample=zeros(nx,ny,num_thin_ctszzs);
    ZZScts_psnr_run=zeros(num_thin_ctszzs+1);
    ZZScts_psnr_run[1] = psnr(x_init, f0_vec)
    X_store=x_init;
    V_store=copy(v_init);
    ZZScts_running_mean= zeros(nxi,num_thin_ctszzs+1);
    ZZScts_running_mean[:,1]= x_init;
    ZZScts_mse_run=zeros(num_thin_ctszzs+1);
    ZZScts_mse_run[1]  =mse(x_init,f0_vec);
    ZZScts_running_var= zeros(nxi,num_thin_ctszzs+1);
    # draw_event_times_lipschitzgrad(x,v) = draw_event_times_lipschitzgrad(x,v,gradU,L)
    give_coeffs_events_lipschitzgrad(x,v) = give_coeffs_events_lipschitzgrad(x,v,gradU,L)
    times = zeros(nxi,num_thin_ctszzs+1);
    for thin_it=1:num_thin_ctszzs
        ZZScts_skeleton=ZigZag(gradU,give_coeffs_events_lipschitzgrad,N, nxi, X_store,V_store)
        times[thin_it+1] = ZZScts_skeleton[end].time
        global X_store=copy((ZZScts_skeleton[end]).position)
        global V_store=Int.((ZZScts_skeleton[end]).velocity)
        ZZScts_sample[:,:,thin_it]=reshape(X_store,size(f0))
        ZZScts_running_mean[:,thin_it+1] = (sum(times[1:thin_it]) / sum(times[1:thin_it+1]))  * ZZScts_running_mean[:,thin_it]  +  (ZZScts_skeleton[end].time / sum(times[1:thin_it+1]) ) * estimate_mean_cts(ZZScts_skeleton)
        ZZScts_psnr_run[thin_it+1]=psnr(ZZScts_running_mean[:,thin_it+1], f0_vec)
        ZZScts_mse_run[thin_it+1]=mse(ZZScts_running_mean[:,thin_it+1], f0_vec)
        # ZZS_running_var[:,thin_it+1]=ZZS_running_var[:,thin_it] + ((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it+1]).*((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it])
        print("ZZS Simulation progress: ", floor(thin_it/num_thin_ctszzs*100), "% \r")
    end
    # ZZS_running_var=ZZS_running_var[:,3:end]./repeat((1:num_thin-1)', nxi,1);
    # ZZS_running_sd=sqrt.(abs.(ZZS_running_var));
    
    println("The PSNR between the ZZS reconstruction and the true image is ", psnr(ZZScts_running_mean[:,end], f0_vec))
    
    save_object(string(respath,"/ZZScts_samples"), ZZScts_sample)
    save_object(string(respath,"/ZZScts_running_mean"), ZZScts_running_mean)
    save_object(string(respath,"/ZZScts_psnr_run"), ZZScts_psnr_run)
    save_object(string(respath,"/ZZScts_mse_run"), ZZScts_mse_run)
    end



# Run Zig Zag

if run_ZZS

delta_ZZS = factor_ZZS * delta_max;
v_init = rand(Bernoulli(0.5),size(x_init));
v_init=Int.(v_init);
v_init=2*v_init .-1;
ZZS_sample=zeros(nx,ny,num_thin);
ZZS_psnr_run=zeros(num_thin+1);
ZZS_psnr_run[1] = psnr(x_init, f0_vec)
X_store=x_init;
V_store=copy(v_init);
ZZS_running_mean= zeros(nxi,num_thin+1);
ZZS_running_mean[:,1]= x_init;
ZZS_mse_run=zeros(num_thin+1);
ZZS_mse_run[1]  =mse(x_init,f0_vec);
ZZS_running_var= zeros(nxi,num_thin+1);

for thin_it=1:num_thin_cts
    ZZS_DBD_skeleton=splitting_zzs_DBD(gradU,delta_ZZS,N,X_store,V_store)
    global X_store=copy((ZZS_DBD_skeleton[end]).position)
    global V_store=Int.((ZZS_DBD_skeleton[end]).velocity)
    ZZS_sample[:,:,thin_it]=reshape(X_store,size(f0))
    ZZS_running_mean[:,thin_it+1] = ZZS_running_mean[:,thin_it] + ((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it])/(thin_it+1)
    ZZS_psnr_run[thin_it+1]=psnr(ZZS_running_mean[:,thin_it+1], f0_vec)
    ZZS_mse_run[thin_it+1]=mse(ZZS_running_mean[:,thin_it+1], f0_vec)
    ZZS_running_var[:,thin_it+1]=ZZS_running_var[:,thin_it] + ((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it+1]).*((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it])
    print("ZZS Simulation progress: ", floor(thin_it/num_thin*100), "% \r")
end

ZZS_running_var=ZZS_running_var[:,3:end]./repeat((1:num_thin-1)', nxi,1);
ZZS_running_sd=sqrt.(abs.(ZZS_running_var));

println("The PSNR between the ZZS (with factor $factor_ZZS) reconstruction and the true image is ", psnr(ZZS_running_mean[:,end], f0_vec))

save_object(string(respath,"/ZZS_samples"), ZZS_sample)
save_object(string(respath,"/ZZS_running_mean"), ZZS_running_mean)
save_object(string(respath,"/ZZS_psnr_run"), ZZS_psnr_run)
save_object(string(respath,"/ZZS_mse_run"), ZZS_mse_run)

end


### BPS

if run_BPS
# BPS_delta_frac=1000;
# delta=BPS_delta_frac*delta_max;
delta_BPS = factor_BPS * delta_max
BPS_sample=zeros(nx,ny,num_thin);
BPS_psnr_run=zeros(num_thin+1);
BPS_mse_run=zeros(num_thin+1);
X_store=copy(x_init);
v_init = vec(randn(size(x_init)));
V_store=copy(v_init);
BPS_running_mean= zeros(nxi,num_thin+1);
BPS_running_mean[:,1]= x_init;
BPS_psnr_run[1]  =psnr(x_init,f0_vec);
BPS_mse_run=zeros(num_thin+1);
BPS_mse_run[1]  =mse(x_init,f0_vec);
BPS_running_var=zeros(nxi,num_thin+1);


for thin_it=1:num_thin
    BPS_RDBDR_skeleton=splitting_bps_RDBDR_gauss(gradU,delta_BPS,N,X_store,V_store)
    global X_store=copy((BPS_RDBDR_skeleton[end]).position)
    global V_store=(BPS_RDBDR_skeleton[end]).velocity
    BPS_sample[:,:,thin_it]=reshape(X_store,size(f0))
    BPS_running_mean[:,thin_it+1] = BPS_running_mean[:,thin_it] + ((BPS_RDBDR_skeleton[end]).position-BPS_running_mean[:,thin_it])/(thin_it+1)
    BPS_psnr_run[thin_it+1]=psnr(BPS_running_mean[:,thin_it+1], f0_vec)
    BPS_mse_run[thin_it+1]=mse(BPS_running_mean[:,thin_it+1], f0_vec)
    BPS_running_var[:,thin_it+1]=BPS_running_var[:,thin_it] + ((BPS_RDBDR_skeleton[end]).position-BPS_running_mean[:,thin_it+1]).*((BPS_RDBDR_skeleton[end]).position-BPS_running_mean[:,thin_it])
    print("BPS Simulation progress: ", floor(thin_it/num_thin*100), "% \r")
end


BPS_running_var=BPS_running_var[:,3:end]./repeat((1:num_thin-1)', nxi,1);
BPS_running_sd=sqrt.(abs.(BPS_running_var));

println("The PSNR between the BPS reconstruction (with factor $factor_BPS and RR $refresh_rate) and the true image is ", psnr(BPS_running_mean[:,end], f0_vec))

save_object(string(respath,"/BPS_samples"), BPS_sample)
save_object(string(respath,"/BPS_running_mean"), BPS_running_mean)
save_object(string(respath,"/BPS_psnr_run"), BPS_psnr_run)

end

### ULA

if run_ULA

ULA_sample=zeros(nx,ny,num_thin);
ULA_psnr_run=zeros(num_thin+1);
X_ULA=copy(x_init);
# println("Initial psnr is ", psnr(X_ULA,f0_vec))
ULA_running_mean= zeros(nxi,num_thin+1);
ULA_running_mean[:,1]= copy(x_init);
M_ULA=x_init;
delta_ULA = factor_ULA * delta_max;
ULA_psnr_run[1]=psnr(ULA_running_mean[:,1], f0_vec)
ULA_mse_run=zeros(num_thin+1);
ULA_mse_run[1]=mse(ULA_running_mean[:,1], f0_vec)
ULA_running_var=zeros(nxi,num_thin+1);

for thin_it=1:num_thin
    global X_ULA=ULA_sim(gradU, delta_ULA,N,X_ULA);
    ULA_sample[:,:,thin_it]=reshape(X_ULA,size(f0));
    ULA_running_mean[:,thin_it+1] = ULA_running_mean[:,thin_it] + (X_ULA-ULA_running_mean[:,thin_it])/(thin_it+1);
    ULA_psnr_run[thin_it+1]=psnr(ULA_running_mean[:,thin_it+1], f0_vec)
    ULA_mse_run[thin_it+1]=mse(ULA_running_mean[:,thin_it+1], f0_vec)
    ULA_running_var[:,thin_it+1]=ULA_running_var[:,thin_it] + (X_ULA-ULA_running_mean[:,thin_it+1]).*(X_ULA-ULA_running_mean[:,thin_it])
    print("ULA Simulation progress: ", floor(thin_it/num_thin*100), "% \r")
end

ULA_mean=(sum(ULA_sample,dims=3))/num_thin;
#println(psnr(ULA_mean,f0))

ULA_running_var=ULA_running_var[:,3:end]./repeat((1:num_thin-1)', nxi,1);
ULA_running_sd=sqrt.(ULA_running_var);

println("The PSNR between the ULA reconstruction (with factor $factor_ULA) and the true image is ", psnr(ULA_running_mean[:,end], f0_vec))

save_object(string(respath,"/ULA_samples"), ULA_sample)
save_object(string(respath,"/ULA_running_mean"), ULA_running_mean)
save_object(string(respath,"/ULA_psnr_run"), ULA_psnr_run)
save_object(string(respath,"/ULA_mse_run"), ULA_mse_run)
end


ZZS_running_mean = load_object(string(respath,"/ZZS_running_mean"))
# ZZScts_running_mean = load_object(string(respath,"/ZZScts_running_mean"))
ULA_running_mean = load_object(string(respath,"/ULA_running_mean"))
# ZZS_mse_run = load(string(respath,"/ZZS_running_mean"))
# ZZScts_mse_run = load(string(respath,"/ZZScts_running_mean"))
# ULA_mse_run = load(string(respath,"/ULA_running_mean"))
ZZS_mse_run = compute_mse_given_mean(ZZS_running_mean, f0_vec, num_thin)
ULA_mse_run = compute_mse_given_mean(ULA_running_mean, f0_vec, num_thin)


# ZZS_running_mean = load(string(respath,"/ZZS_running_mean"))
# ULA_running_mean = load(string(respath,"/ULA_running_mean"))

if run_BPS == true
    plot((0:num_thin)*N,ZZS_psnr_run,label="ZZS",
        xlabel = "Iterations",
        ylabel = "PSNR",
        legendfontsize=10,
        legend=:bottomright,
        linewidth = 2,
        color = "red"
        )
plot!((0:num_thin)*N,BPS_psnr_run,label="BPS",linewidth = 1, color = "orange")
plot!((0:num_thin)*N,ULA_psnr_run,label="ULA",linewidth = 2, color = "green")
p_psnr=hline!([s_psnr[ind]],label="Posterior mean", color = "black",linewidth = 1.5,line=:dash)
display(p_psnr)
savefig(p_psnr, string(respath,"/psnr_tv_",name,"_blur_", blur_length,".pdf"))


p_ZZS=plot(Gray.(reshape(ZZS_running_mean[:,end],size(f0))),title="ZZS",axis=([], false))
p_BPS=plot(Gray.(reshape(BPS_running_mean[:,end],size(f0))),title="BPS",axis=([], false))
p_ULA=plot(Gray.(reshape(ULA_running_mean[:,end],size(f0))),title="ULA",axis=([], false))

plot_recon=plot(p_true,p_noise,p_ZZS,p_BPS,p_ULA, layout=6)
display(plot_recon)
savefig(plot_recon, string(respath,"/recon_tv_",name,"_blur_", blur_length,".pdf"))

p_ZZS_low = plot(Gray.(reshape(ZZS_running_mean[:,thin_low],size(f0))),title="ZZS",axis=([], false))
p_ULA_low = plot(Gray.(reshape(ULA_running_mean[:,thin_low],size(f0))),title="ULA",axis=([], false))
p_BPS_low = plot(Gray.(reshape(BPS_running_mean[:,thin_low],size(f0))),title="BPS",axis=([], false))
plot_recon_low=plot(p_true,p_noise,p_ZZS_low,p_BPS_low,p_ULA_low, layout=6)
display(plot_recon_low)
savefig(plot_recon_low, string(respath,"/recon_low_tv_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))

plot_recon_all=plot(p_true,p_noise,p_L2,p_ZZS_low,p_BPS_low,p_ULA_low,p_ZZS,p_BPS,p_ULA,layout=9,titlefontsize=10, margin = .0mm)
display(plot_recon_all)
savefig(plot_recon_all, string(respath,"/recon_all_tv_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))

    ## MSE plot
    plot((0:num_thin)*N,ZZS_mse_run,label="ZZS",
            xlabel = "Iterations",
            ylabel = "MSE",
            legendfontsize=10,
            legend=:topright,
            linewidth = 2,
            yaxis=:log,
            color = "red",
            # ylim = [0.027,1]
            )
            
    plot!((0:num_thin)*N,BPS_mse_run,label="BPS",linewidth = 1, color = "orange")
    p_mse = plot!((0:num_thin)*N,ULA_mse_run,label="ULA",linewidth = 2, color = "green")
    display(p_mse)
    savefig(p_mse, string(respath,"/mse_tv_",name,"_blur_", blur_length,".pdf"))

else
#     plot((0:num_thin)*N,ZZS_psnr_run,label="ZZS",
#     xlabel = "Iterations",
#     ylabel = "PSNR",
#     legendfontsize=10,
#     legend=:bottomright,
#     linewidth = 2,
#     color = "red"
#     )
# p_psnr = plot!((0:num_thin)*N,ULA_psnr_run,label="ULA",linewidth = 2, color = "green")
# display(p_psnr)
# savefig(p_psnr, string(respath,"/psnr_tv_",name,"_blur_", blur_length,".pdf"))
plot_recon=plot(p_true,p_noise, layout=(2,1))
display(plot_recon)
savefig(plot_recon, string(respath,"/trueandobserved_tv_",name,"_blur_", blur_length,".pdf"))


p_ZZS=plot(Gray.(reshape(ZZS_running_mean[:,end],size(f0))),title="ZZS",axis=([], false))
p_ZZScont = plot(Gray.(reshape(ZZScts_running_mean[:,end],size(f0))),title="ZZS (cont.)",axis=([], false))
p_ULA=plot(Gray.(reshape(ULA_running_mean[:,end],size(f0))),title="ULA",axis=([], false))
p_ZZS_low = plot(Gray.(reshape(ZZS_running_mean[:,thin_low],size(f0))),title="ZZS",axis=([], false))
p_ZZScontlow = plot(Gray.(reshape(ZZScts_running_mean[:,thin_low],size(f0))),title="ZZS (cont.)",axis=([], false))
p_ULA_low = plot(Gray.(reshape(ULA_running_mean[:,thin_low],size(f0))),title="ULA",axis=([], false))


plot_recon_low=plot(p_ZZS_low,p_ULA_low,p_ZZScontlow)
display(plot_recon_low)
savefig(plot_recon_low, string(respath,"/reconlow_tv_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))

plot_recon_allrecons=plot(p_ZZS_low,p_ULA_low,p_ZZScontlow,p_ZZS,p_ULA,p_ZZScont, layout=(2,3))
display(plot_recon_allrecons)
savefig(plot_recon_allrecons, string(respath,"/reconall_tv_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))


plot_recon_allrecons=plot(p_ZZS_low,p_ULA_low,p_ZZScontlow,p_ZZS,p_ULA,p_ZZScont, layout=(2,3),titlefontsize=10, margin = .0mm)


# p_ZZS_low = plot(Gray.(reshape(ZZS_running_mean[:,thin_low],size(f0))),title="ZZS",axis=([], false))
# p_ULA_low = plot(Gray.(reshape(ULA_running_mean[:,thin_low],size(f0))),title="ULA",axis=([], false))
# plot_recon_low=plot(p_true,p_noise,p_ZZS_low,p_ULA_low, layout=6)
# display(plot_recon_low)
# savefig(plot_recon_low, string(respath,"/recon_low_tv_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))

plot_recon_all=plot(p_true,p_ZZS_low,p_ULA_low,p_ZZScontlow,p_noise,p_ZZS,p_ULA,p_ZZScont,layout=(2,4),titlefontsize=10, margin = .0mm)
display(plot_recon_all)
savefig(plot_recon_all, string(respath,"/recon_allfigs_tv_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))

## MSE plot
plot((0:num_thin)*N,ZZS_mse_run,label="ZZS",
        xlabel = "Iterations",
        ylabel = "MSE",
        legendfontsize=10,
        legend=:topright,
        linewidth = 2,
        yaxis=:log,
        color = "red",
        xlim = [-5000,2.08*10^5],
        xticks = [0, 5*10^4,10^5,1.5*10^5,2*10^5],
        ylim = [10^(-2.28),10^(-1.5)]
        )
plot!((0:num_thin)*N,ZZScts_mse_run,label="ZZS (continuous)",linewidth = 2, color = "blue")
p_mse = plot!((0:num_thin)*N,ULA_mse_run,label="ULA",linewidth = 2, color = "green")
display(p_mse)
savefig(p_mse, string(respath,"/mse_tv_",name,"_blur_", blur_length,".pdf"))


    # p_ZZS=plot(Gray.(reshape(ZZS_running_mean[:,end],size(f0))),title="ZZS",axis=([], false))
    # p_ULA=plot(Gray.(reshape(ULA_running_mean[:,end],size(f0))),title="ULA",axis=([], false))
    
    # plot_recon=plot(p_true,p_noise,p_ZZS,p_ULA, layout=(2,2))
    # display(plot_recon)
    # savefig(plot_recon, string(respath,"/recon.pdf"))
    
    # plot((0:num_thin)*N,ZZS_psnr_run,label="ZZS",
    #         xlabel = "Iterations",
    #         ylabel = "PSNR",
    #         legendfontsize=10,
    #         legend=:bottomright,
    #         linewidth = 2,
    #         )
    # p_psnr = plot!((0:num_thin)*N,ULA_psnr_run,label="ULA",linewidth = 2,)
    # # p_psnr=hline!([s_psnr[ind]],label="Posterior mean")
    # display(p_psnr)
    
    # ## MSE plot
    # plot((0:num_thin)*N,ZZS_mse_run,label="ZZS",
    #         xlabel = "Iterations",
    #         ylabel = "MSE",
    #         legendfontsize=10,
    #         legend=:topright,
    #         linewidth = 2,
    #         yaxis=:log
    #         )
    # p_mse = plot!((0:num_thin)*N,ULA_mse_run,label="ULA",linewidth = 2,)
    # # p_psnr=hline!([s_psnr[ind]],label="Posterior mean")
    # display(p_mse)
    # savefig(p_mse, string(respath,"/mse.pdf"))

    
    # savefig(p_psnr, string(respath,"/psnr.pdf"))
    
    # p_ZZS_low = plot(Gray.(reshape(ZZS_running_mean[:,thin_low],size(f0))),title="ZZS",axis=([], false))
    # p_ULA_low = plot(Gray.(reshape(ULA_running_mean[:,thin_low],size(f0))),title="ULA",axis=([], false))
    # plot_recon_low=plot(p_true,p_noise,p_ZZS_low,p_ULA_low, layout=(2,2))
    # display(plot_recon_low)
    # savefig(plot_recon_low, string(respath,"/recon_low_", thin_low,".pdf"))
    
    
    # p_ZZS_sd=plot(Gray.(reshape(ZZS_running_sd[:,end],size(f0))),title="ZZS Standard Deviation",axis=([], false))
    # p_BPS_sd=plot(Gray.(reshape(BPS_running_sd[:,end],size(f0))),title="BPS Standard Deviation",axis=([], false))
    # p_ULA_sd=plot(Gray.(reshape(ULA_running_sd[:,end],size(f0))),title="ULA Standard Deviation",axis=([], false))
    
    # plot_sd=plot(p_ZZS_sd,p_ULA_sd, layout=2)
    # display(plot_sd)

end