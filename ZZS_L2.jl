using Plots
using Images
using JLD2  # For saving data
using FFTW  # For fast fourier transform
using Distributions  # For probability distributions
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
respath="results/deblur_L2"
#n = 128;
#name = "Images/cman.png";
#f0 = Float64.(load(name));
#f0=f0[Int(floor(n/2)):Int(floor(3*n/2)),Int(floor(n/2)):Int(floor(3*n/2))]

f0=load_oject(string(respath,"/true_image"))

#p_true=plot(Gray.(f0),title="Ground Truth",axis=([], false))
#display(p_true)
#f0_vec=vec(reshape(f0,:,1))
#(nx,ny)=size(f0);
#blur_length = 9;


#if !isdir(respath)
#    mkdir(respath)
#end

#function cshift(x,L)
#    N=length(x)
#    y = zeros(N)
#
#    if L==0
#        y=x
#    else
#        L=Int(-L)
#        y[1:N-L]=x[L+1:N]
#        y[N-L+1:N]=x[1:L]
#    end
#    return y
#end

function mse(x,y)
    m=sum((x-y).^2)/length(x)
    return m
end

function psnr(x,y)
    m=max(maximum(abs.(x)),maximum(abs.(y)))
    p=20*log10(m)-10*log10(mse(x,y))
    return p
end

#h=ones(blur_length);
#lh=length(h);
#h = h/sum(h);
#h=vcat(h, zeros(nx-lh));
#h=cshift(h,-(lh-1)/2);
#h = reshape(h,:,1)*reshape(h,1,:);
#hF =fft(h);

#hF_vec=vec(reshape(h,:,1));
#hFC=hF';

#y=real(ifft(fft(f0).*hF));
#y_vec=vec(reshape(y,:,1))

#sigma=1/255;
#sigma2=sigma*sigma;
#y=y+sigma*randn(nx,ny);
#p_noise=plot(Gray.(y), size=(256,256),title="Observed Image",axis=([], false))

y=load_object(string(respath,"/noisy_image"))

y_vec=vec(reshape(y,:,1))

yF=fft(y);
yF_vec=vec(reshape(yF,:,1));

#s_set=exp10.(range(-5,-1,length=1000));
#s_psnr=zeros(length(s_set));

#for s_it in eachindex(s_set)
#    s=s_set[s_it];
#    X=real(ifft(yF.*hFC./(abs.(hF).^2 .+s)));
#    s_psnr[s_it]=psnr(f0,X)
#end
#display(plot(s_set,s_psnr, title="PSNR for different parameters", xlabel="s"))
#ind=argmax(s_psnr)
#println("We have tuned s=", s_set[ind], " at this value the PSNR is ", s_psnr[ind])
#s_max=s_set[ind];

#post_mean=real(ifft(yF.*hFC./(abs.(hFC).^2 .+s_max)))
#p_L2=plot(Gray.(post_mean),title="Posterior Mean",axis=([], false))

## Likelihood

A=x->reshape(real(ifft(fft(reshape(x,size(f0))).*hF)),:,1);
AT=x->reshape(real(ifft(fft(reshape(x,size(f0))).*hFC)),:,1);



# fLike=x->norm(A(x)-y)^2/(2*sigma2);
#yFf=vec(reshape(yF,:,1));
y_vec=vec(reshape(y,:,1));
gradf = x-> AT(A(x)-y_vec)/sigma2
##prior
# g=x->s/sigma2*norm(x)^2;
s_max=load_object(string(respath,"/s"))
gradg = x-> s_max*x/sigma2

# Run Zig Zag

gradU=x->vec(gradf(x)+gradg(x))
delta=sigma2;
N=10^3;   #Number of steps per sample we store
x_init=vec(reshape(y, :,1));
nxi=length(x_init)
v_init = rand(Bernoulli(0.5),size(x_init));
v_init=Int.(v_init);
v_init=2*v_init .-1;
num_thin=10^3;    # Number of samples we store
ZZS_sample=zeros(nx,ny,num_thin);
ZZS_psnr_run=zeros(num_thin);
X_store=x_init;
V_store=copy(v_init);
ZZS_running_mean= zeros(nxi,num_thin+1);
ZZS_running_mean[:,1]= copy(x_init);

for thin_it=1:num_thin
    ZZS_DBD_skeleton=splitting_zzs_DBD(gradU,delta,N,X_store,V_store)
    global X_store=copy((ZZS_DBD_skeleton[end]).position)
    global V_store=Int.((ZZS_DBD_skeleton[end]).velocity)
    ZZS_sample[:,:,thin_it]=reshape(X_store,size(f0))
    ZZS_running_mean[:,thin_it+1] = ZZS_running_mean[:,thin_it] + ((ZZS_DBD_skeleton[end]).position-ZZS_running_mean[:,thin_it])/thin_it
    ZZS_psnr_run[thin_it]=psnr(ZZS_running_mean[:,thin_it], f0_vec)
    print("ZZS Simulation progress: ", floor(thin_it/num_thin*100), "% \r")
end

println("The PSNR between the ZZS reconstruction and the true image is ", psnr(ZZS_running_mean[:,end], f0_vec))

save_object(string(respath,"/ZZS_samples_delta_1e_5"), ZZS_sample)
save_object(string(respath,"/ZZS_running_mean_delta_1e_5"), ZZS_running_mean)
save_object(string(respath,"/ZZS_psnr_run_delta_1e_5"), ZZS_psnr_run)


p_ZZS=plot(Gray.(reshape(ZZS_running_mean[:,end],size(f0))),title="ZZS reconstruction",axis=([], false))
p_BPS=plot(Gray.(reshape(BPS_running_mean[:,end],size(f0))),title="BPS reconstruction",axis=([], false))
p_ULA=plot(Gray.(reshape(ULA_running_mean[:,end],size(f0))),title="ULA reconstruction",axis=([], false))

plot_recon=plot(p_true,p_noise,p_L2,p_ZZS,p_BPS,p_ULA, layout=6)

display(plot_recon)
savefig(plot_recon, string(respath,"/recon.pdf"))
