using Plots
using Images
using JLD2  # For saving data

respath="results/deblur_L2"

ZZS_sample = load_object(string(respath,"/ZZS_samples"))
ZZS_running_mean = load_object(string(respath,"/ZZS_running_mean"))
ZZS_psnr_run = load_object(string(respath,"/ZZS_psnr_run"))
ULA_sample = load_object(string(respath,"/ULA_samples"))
ULA_running_mean = load_object(string(respath,"/ULA_running_mean"))
ULA_psnr_run = load_object(string(respath,"/ULA_psnr_run"))

p_ZZS_psnr=plot(ZZS_psnr_run,label="ZZS")
p_ULA_psnr=plot(ULA_psnr_run, label="ULA")
display(plot(p_ULA_psnr,p_ZZS_psnr))

p_ZZS=plot(Gray.(reshape(ZZS_running_mean[:,end],size(f0))),title="ZZS reconstruction")
p_ULA=plot(Gray.(reshape(ULA_running_mean[:,end],size(f0))),title="ULA reconstruction")
#display(plot(p_true,p_noise,p_L2,p_ZZS,p_ULA, layout=5))
display(plot(p_ZZS,p_ULA, layout=2))


num_thin=100
display(plot([1:num_thin,1:num_thin],[ZZS_psnr_run,ULA_psnr_run],label=["ZZS","ULA"]))
