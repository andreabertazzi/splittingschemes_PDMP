using Plots
using Images
using JLD2           # For saving data
using FFTW           # For fast fourier transform
using Distributions  # For probability distributions
using Random         # To use same seed to generate random numbers when needed
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")

run_ZZS = true
run_BPS = true
run_ULA = true
factor_ZZS = 225
factor_BPS = 500
factor_ULA = 1.98
refresh_rate = 1e-2;
thin_low = 100

N=10^3;   #Number of steps per sample we store
num_thin = 10*10^2;    # Number of samples we store
# num_thin = 5*10^1;    # Number of samples we store
# num_thin = 5*10^0; 
refresh_rate = 1e-2;
# BSNRdb = 50;
BSNRdb = false
blur_length = 31;

# name = "cman";
# name = "handwritten";
name = "cmansmall";
f0 = Float64.(load(string("Images/",name,".png")));


# if maximum(maximum(f0))>1
#     f0=f0/255
# end

println("The maximum pixel is ", maximum(maximum(f0)))
#f0 = Float64.(load(name));
#f0=f0[Int(floor(n/2)):Int(floor(3*n/2)),Int(floor(n/2)):Int(floor(3*n/2))]



p_true=plot(Gray.(f0),title="Ground Truth",axis=([], false))
#display(p_true)
f0_vec=vec(reshape(f0,:,1))
(nx,ny)=size(f0);

respath="results/deblur_L2"
if !isdir(respath)
    mkdir(respath)
end

save_object(string(respath,"/true_image"), f0)

function cshift(x,L)
    N=length(x)
    y = zeros(N)

    if L==0
        y=x
    else
        L=Int(-L)
        y[1:N-L]=x[L+1:N]
        y[N-L+1:N]=x[1:L]
    end
    return y
end

function mse(x,y)
    m=sum((x-y).^2)/length(x)
    return m
end

function psnr(x,y)
    m=max(maximum(abs.(x)),maximum(abs.(y)))
    p=20*log10(m)-10*log10(mse(x,y))
    return p
end

h=ones(blur_length);
lh=length(h);
h = h/sum(h);
h=vcat(h, zeros(nx-lh));
h=cshift(h,-(lh-1)/2);
h = reshape(h,:,1)*reshape(h,1,:);
hF =fft(h);

hF_vec=vec(reshape(h,:,1));
hFC=hF';

# display(plot(sort(vec(abs.(hF)^2)), seriestype=:scatter, title="Eigenvalues of Blur operator"))

y=real(ifft(fft(f0).*hF));
y_vec=vec(reshape(y,:,1))


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
Random.seed!()

save_object(string(respath,"/noisy_image"), y)

p_noise=plot(Gray.(y), title="Observed Image",axis=([], false))
# display(p_noise)

y_vec=vec(reshape(y,:,1))

yF=fft(y);
yF_vec=vec(reshape(yF,:,1));

# For the prior we use a Gaussian prior with variance sigma2/s
# We will tune s by doing a grid search and selecting the optimal psnr with respect to the truth

s_set=exp10.(range(-5,-1,length=1000));
s_psnr=zeros(length(s_set));

for s_it in eachindex(s_set)
    s=s_set[s_it];
    X=real(ifft(yF.*hFC./(abs.(hF).^2 .+s)));
    s_psnr[s_it]=psnr(f0,X)
end
#display(plot(s_set,s_psnr, title="PSNR for different parameters", xlabel="s"))
ind=argmax(s_psnr)
println("We have tuned s=", s_set[ind], " at this value the PSNR is ", s_psnr[ind])
s_max=s_set[ind];
save_object(string(respath,"/s"), s_max)
post_mean=real(ifft(yF.*hFC./(abs.(hFC).^2 .+s_max)))
p_L2=plot(Gray.(post_mean),title="Posterior Mean",axis=([], false))

## Likelihood

A=x->reshape(real(ifft(fft(reshape(x,size(f0))).*hF)),:,1);
AT=x->reshape(real(ifft(fft(reshape(x,size(f0))).*hFC)),:,1);



# fLike=x->norm(A(x)-y)^2/(2*sigma2);
yFf=vec(reshape(yF,:,1));
y_vec=vec(reshape(y,:,1));
gradf = x-> AT(A(x)-y_vec)/sigma2
##prior
# g=x->s/sigma2*norm(x)^2;
gradg = x-> s_max*x/sigma2

# Stability Barrier
L=1/sigma2+s_max/sigma2;
delta_base=1/L;


# Run Zig Zag

gradU=x->vec(gradf(x)+gradg(x))
x_init=vec(reshape(y, :,1));

if run_ZZS

delta_ZZS = factor_ZZS * delta_base;
nxi=length(x_init)
v_init = rand(Bernoulli(0.5),size(x_init));
v_init=Int.(v_init);
v_init=2*v_init .-1;
ZZS_sample=zeros(nx,ny,num_thin);
# ZZS_psnr_run=zeros(num_thin);
ZZS_psnr_run=zeros(num_thin+1);
ZZS_psnr_run[1] = psnr(x_init, f0_vec)
X_store=x_init;
V_store=copy(v_init);
ZZS_running_mean= zeros(nxi,num_thin+1);
ZZS_running_mean[:,1]= copy(x_init);
ZZS_running_var= zeros(nxi,num_thin+1);
ZZS_mse_run=zeros(num_thin+1);
ZZS_mse_run[1]  =mse(x_init,f0_vec);

for thin_it=1:num_thin
    ZZS_DBD_skeleton=splitting_zzs_DBD(gradU,delta_ZZS,N,X_store,V_store)
    global X_store=copy((ZZS_DBD_skeleton[end]).position)
    global V_store=Int.((ZZS_DBD_skeleton[end]).velocity)
    ZZS_sample[:,:,thin_it]=reshape(X_store,size(f0))
    ZZS_running_mean[:,thin_it+1] = ZZS_running_mean[:,thin_it] + ((ZZS_DBD_skeleton[end]).position - ZZS_running_mean[:,thin_it])/(thin_it+1)
    ZZS_psnr_run[thin_it+1]=psnr(ZZS_running_mean[:,thin_it+1], f0_vec)
    ZZS_mse_run[thin_it+1]=mse(ZZS_running_mean[:,thin_it+1], f0_vec)
    ZZS_running_var[:,thin_it+1]=ZZS_running_var[:,thin_it] + ((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it+1]).*((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it])
    print("ZZS Simulation progress: ", floor(thin_it/num_thin*100), "% \r")
end

ZZS_running_var=ZZS_running_var[:,3:end]./repeat((1:num_thin-1)', nxi,1);
ZZS_running_sd=sqrt.(abs.(ZZS_running_var));

println("The PSNR between the ZZS reconstruction (with factor $factor_ZZS) and the true image is ", psnr(ZZS_running_mean[:,end], f0_vec))

save_object(string(respath,"/ZZS_samples_delta_1e_5"), ZZS_sample)
save_object(string(respath,"/ZZS_running_mean_delta_1e_5"), ZZS_running_mean)
save_object(string(respath,"/ZZS_psnr_run_delta_1e_5"), ZZS_psnr_run)

end


### BPS

if run_BPS

# BPS_delta_frac=10;
delta_BPS = factor_BPS * delta_base;
BPS_sample=zeros(nx,ny,num_thin);
BPS_psnr_run=zeros(num_thin+1);
BPS_psnr_run[1] = psnr(x_init, f0_vec)
X_store=x_init;
v_init  = vec(randn(size(x_init)));
V_store = copy(v_init);
BPS_running_mean= zeros(nxi,num_thin+1);
BPS_running_mean[:,1]= x_init;
BPS_running_var=zeros(nxi,num_thin+1);
BPS_mse_run=zeros(num_thin+1);
BPS_mse_run[1]  =mse(x_init,f0_vec);

for thin_it=1:num_thin
    BPS_RDBDR_skeleton=splitting_bps_RDBDR_gauss(gradU,delta_BPS,N,X_store,V_store)
    global X_store=copy((BPS_RDBDR_skeleton[end]).position)
    global V_store=(BPS_RDBDR_skeleton[end]).velocity
    BPS_sample[:,:,thin_it]=reshape(X_store,size(f0))
    BPS_running_mean[:,thin_it+1] = BPS_running_mean[:,thin_it] + ((BPS_RDBDR_skeleton[end]).position - BPS_running_mean[:,thin_it])/(thin_it+1)
    BPS_psnr_run[thin_it+1]=psnr(BPS_running_mean[:,thin_it+1], f0_vec)
    BPS_mse_run[thin_it+1]=mse(BPS_running_mean[:,thin_it+1], f0_vec)
    BPS_running_var[:,thin_it+1]=BPS_running_var[:,thin_it] + ((BPS_RDBDR_skeleton[end]).position-BPS_running_mean[:,thin_it+1]).*((BPS_RDBDR_skeleton[end]).position-BPS_running_mean[:,thin_it])
    print("BPS Simulation progress: ", floor(thin_it/num_thin*100), "% \r")
end

println("The PSNR between the BPS reconstruction (with factor $factor_BPS and RR $refresh_rate) and the true image is ", psnr(BPS_running_mean[:,end], f0_vec))


BPS_running_var=BPS_running_var[:,3:end]./repeat((1:num_thin-1)', nxi,1);
BPS_running_sd=sqrt.(abs.(BPS_running_var));


save_object(string(respath,"/BPS_samples"), BPS_sample)
save_object(string(respath,"/BPS_running_mean"), BPS_running_mean)
save_object(string(respath,"/BPS_psnr_run"), BPS_psnr_run)

end



### ULA

if run_ULA
nxi=length(x_init)
delta_ULA= factor_ULA * delta_base;
ULA_sample=zeros(nx,ny,num_thin);
# ULA_psnr_run=zeros(num_thin);
ULA_psnr_run=zeros(num_thin+1);
ULA_psnr_run[1] = psnr(x_init, f0_vec)
X_ULA=copy(x_init);
ULA_running_mean= zeros(nxi,num_thin+1);
ULA_running_mean[:,1]= x_init;
ULA_running_var=zeros(nxi,num_thin+1);
M_ULA=x_init;
ULA_mse_run=zeros(num_thin+1);
ULA_mse_run[1]=mse(ULA_running_mean[:,1], f0_vec)

for thin_it=1:num_thin
    global X_ULA=ULA_sim(gradU, delta_ULA,N,X_ULA);
    ULA_sample[:,:,thin_it]=reshape(X_ULA,size(f0));
    ULA_running_mean[:,thin_it+1] = ULA_running_mean[:,thin_it] + (X_ULA - ULA_running_mean[:,thin_it])/(thin_it+1)
    ULA_psnr_run[thin_it+1]=psnr(ULA_running_mean[:,thin_it+1], f0_vec)
    # ULA_running_mean[:,thin_it+1] = ULA_running_mean[:,thin_it] + (X_ULA-ULA_running_mean[:,thin_it])/thin_it;
    # ULA_psnr_run[thin_it+1]=psnr(ULA_running_mean[:,thin_it], f0_vec)
    ULA_mse_run[thin_it+1]=mse(ULA_running_mean[:,thin_it+1], f0_vec)
    ULA_running_var[:,thin_it+1]=ULA_running_var[:,thin_it] + (X_ULA-ULA_running_mean[:,thin_it+1]).*(X_ULA-ULA_running_mean[:,thin_it])
    print("ULA Simulation progress: ", floor(thin_it/num_thin*100), "% \r")
end

println("The PSNR between the ULA reconstruction (with factor $factor_ULA) and the true image is ", psnr(ULA_running_mean[:,end], f0_vec))


ULA_running_var=ULA_running_var[:,3:end]./repeat((1:num_thin-1)', nxi,1);
ULA_running_sd=sqrt.(abs.(ULA_running_var));

save_object(string(respath,"/ULA_samples"), ULA_sample)
save_object(string(respath,"/ULA_running_mean"), ULA_running_mean)
save_object(string(respath,"/ULA_psnr_run"), ULA_psnr_run)

end


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
savefig(p_psnr, string(respath,"/psnr_gauss_",name,"_blur_", blur_length,".pdf"))


p_ZZS=plot(Gray.(reshape(ZZS_running_mean[:,end],size(f0))),title="ZZS",axis=([], false))
p_BPS=plot(Gray.(reshape(BPS_running_mean[:,end],size(f0))),title="BPS",axis=([], false))
p_ULA=plot(Gray.(reshape(ULA_running_mean[:,end],size(f0))),title="ULA",axis=([], false))

plot_recon=plot(p_true,p_noise,p_L2,p_ZZS,p_BPS,p_ULA, layout=6)
display(plot_recon)
savefig(plot_recon, string(respath,"/recon_gauss_",name,"_blur_", blur_length,".pdf"))

p_ZZS_low = plot(Gray.(reshape(ZZS_running_mean[:,thin_low],size(f0))),title="ZZS",axis=([], false))
p_ULA_low = plot(Gray.(reshape(ULA_running_mean[:,thin_low],size(f0))),title="ULA",axis=([], false))
p_BPS_low = plot(Gray.(reshape(BPS_running_mean[:,thin_low],size(f0))),title="BPS",axis=([], false))
plot_recon_low=plot(p_true,p_noise,p_L2,p_ZZS_low,p_BPS_low,p_ULA_low, layout=6)
display(plot_recon_low)
savefig(plot_recon_low, string(respath,"/recon_low_gauss_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))

plot_recon_all=plot(p_true,p_noise,p_L2,p_ZZS_low,p_BPS_low,p_ULA_low,p_ZZS,p_BPS,p_ULA,layout=9,titlefontsize=10, margin = .0mm)
display(plot_recon_all)
savefig(plot_recon_all, string(respath,"/recon_all_gauss_",name,"_blur_", blur_length,"_thin_",thin_low,".pdf"))

    ## MSE plot
    plot((0:num_thin)*N,ZZS_mse_run,label="ZZS",
            xlabel = "Iterations",
            ylabel = "MSE",
            legendfontsize=10,
            legend=:topright,
            linewidth = 2,
            yaxis=:log,
            color = "red",
            ylim = [0.027,1]
            )
            
    plot!((0:num_thin)*N,BPS_mse_run,label="BPS",linewidth = 1, color = "orange")
    p_mse = plot!((0:num_thin)*N,ULA_mse_run,label="ULA",linewidth = 2, color = "green")
    display(p_mse)
    savefig(p_mse, string(respath,"/mse_gauss_",name,"_blur_", blur_length,".pdf"))
# p_ZZS_sd=plot(Gray.(reshape(ZZS_running_sd[:,end],size(f0))),title="ZZS Standard Deviation",axis=([], false))
# p_BPS_sd=plot(Gray.(reshape(BPS_running_sd[:,end],size(f0))),title="BPS Standard Deviation",axis=([], false))
# p_ULA_sd=plot(Gray.(reshape(ULA_running_sd[:,end],size(f0))),title="ULA Standard Deviation",axis=([], false))

# plot_sd=plot(p_ZZS_sd,p_BPS_sd,p_ULA_sd, layout=3)
# display(plot_sd)
