include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("run_experiments.jl")

## For Gaussian target
# dim = 20
# ρ = 0.
# Σ = ρ * ones(dim, dim) + (1 - ρ)I
# # Σ[1,1] = 0.05
# Σ_inv = inv(Σ)
# μ = zeros(dim)
# ∇U(x) = Σ_inv * (x - μ)
# target(x) =  exp(-dot(x-μ,Σ_inv*(x-μ))/2)  # sufficient to have it unnormalised
# # true_mean = 0.0 # we only consider the first component of the mean
# true_rad = Float64(sum(diag(Σ)))
# truth = [0,true_rad]
# mean_fn(x) = x[1]
# radius(x) = sum(x.^2)
# test_funcs = Vector{Function}(undef,2)
# test_funcs[1] = mean_fn
# test_funcs[2] = radius

## If want to count the number of rejections and make nice plots, run this

δ = [0.3]
# δ = [0.05*i for i=1:20]
# N = Integer(round(T/δ))
N = 10^3
# n_exp = 100
n_exp = 100
refresh_rate = 0.5
# dims = [1,2,3,4,5,6,7,8,9,10,12,15,17,20,25,30,40,50,75,100]
# dims = [1, 3, 5, 8, 12, 15, 20, 28, 35, 50, 65, 80, 150, 200,250,350,500,750,1000,1250,1500,1750,200]
dims = [20]
# dims = [1, 3, 5, 8, 12, 15, 20, 28, 80, 150, 250,500,750,1000,1250,1500,1750,2000,2500]

## For correlated covariance matrix
# correlations = [0.05*i for i=0:18]
# second_bit = [(0.9+0.01*i) for i=1:9]
# correlations=vcat(correlations,second_bit)
# correlations = [0.5]
correlations = [0.0, 0.15, 0.3, 0.45, 0.6, 0.9]
Cov(ρ,d) = ρ * ones(d, d) + (1 - ρ)I
Cov_inv(ρ,d) = inv(Cov(ρ,d))

## For diagonal covariance matrix
# correlations = [0.1]
# correlations = [0.0001*(i) for i=1:40:2001]
# d_half(d) = Integer(ceil(d/2))
# Cov(a,d) = diagm([a*ones(d_half(d));ones(d-d_half(d))])
# Cov_inv(a,d) = diagm([1/a*ones(d_half(d));ones(d-d_half(d))])
# Cov(a,d) = diagm([a;ones(d-1)])
# Cov_inv(a,d) = diagm([1/a;ones(d-1)])
# Cov(a, d) = diagm([a; ones(d - 1)])
# Cov_inv(a, d) = diagm([1 / a; ones(d - 1)])
# Cov(a,d) = diagm([a/d;ones(d-1)])
# Cov_inv(a,d) = diagm([d/a;ones(d-1)])

∇U(x, Σ_inv) = Σ_inv * x
target_unnorm(x, Σ_inv) = exp(-dot(x, Σ_inv * x) / 2)

samplers = [zzs_metropolised, bps_metropolised]
draw_v = [draw_velocity_zzs, draw_velocity_gauss]

n_rej = count_rejections_metropolis(
    samplers,
    draw_v,
    target_unnorm,
    Cov,
    Cov_inv,
    ∇U,
    δ,
    dims,
    correlations,
    n_exp,
    N,
)


## PLOTS

colours = [:red, :blue, :green]

if length(dims) > 1
    n_rej_zzs_percentage = n_rej[1, :, 1, :, 1] / N
    mean_nrej_zzs_percent = transpose(mean(n_rej_zzs_percentage; dims = 1))
    n_rej_bps_percent = n_rej[2, :, 1, :, 1] / N
    mean_nrej_bps_percent = transpose(mean(n_rej_bps_percent; dims = 1))
    # n_rej_zzsgauss_percent = n_rej[3,:,1,:,1]/N
    # mean_nrej_zzsgauss_percent = transpose(mean(n_rej_zzsgauss_percent;dims=1))
    plot(
        dims,
        mean_nrej_zzs_percent,
        ylabel = "Fraction of rejections",
        xlabel = "Dimension",
        linewidth = 2,
        legend = false,
        # legend=:topleft, legendfontsize=10,
        linecolor = colours[1],
        label = "ZZS - Splitting DBD",
        # ylim = [0,0.725],
        # xticks = [1, 50, 100, 150, 200, 250],
        # xticks = [1,50,100,150,200,250,300,350,400,450,500]
    )
    plot!(
        dims,
        mean_nrej_bps_percent,
        linewidth = 2,
        linecolor = colours[2],
        label = "BPS - Splitting RDBDR",
    )
    # savefig(string("n_rejections_bydims_delta_",δ[1],"_corr_",correlations[1],"_nexp_",n_exp,".pdf"))
    # savefig(string("n_rejections_bydims_delta_",δ[1],"_var_",correlations[1],"_nexp_",n_exp,".pdf"))
    # savefig(string("n_rejections_bydims_to_",dims[end],"_delta_",δ[1],"_var_",correlations[1],"_nexp_",n_exp,".pdf"))
elseif length(δ) > 1 # Plot as function of step size
    n_rej_zzs_percentage = n_rej[1, :, 1, 1, :] / N
    mean_nrej_zzs_percent = transpose(mean(n_rej_zzs_percentage; dims = 1))
    n_rej_bps_percent = n_rej[2, :, 1, 1, :] / N
    mean_nrej_bps_percent = transpose(mean(n_rej_bps_percent; dims = 1))
    # n_rej_zzsgauss_percent = n_rej[3,:,1,1,:]/N
    # mean_nrej_zzsgauss_percent = transpose(mean(n_rej_zzsgauss_percent;dims=1))
    plot(
        δ,
        mean_nrej_zzs_percent,
        ylabel = "Fraction of rejections",
        xlabel = "Step size",
        linewidth = 2,
        # legend=:topleft, legendfontsize=10,
        # linewidth = 3,
        legend = false,
        # linestyle=:dash,
        linecolor = colours[1],
        label = "ZZS - Splitting DBD",
    )
    plot(
        δ,
        mean_nrej_bps_percent,
        linewidth = 2,
        linecolor = colours[2],
        label = "BPS - Splitting RDBDR",
    )

    # savefig(string("n_rejections_bydelta_nolegend_var",correlations[1],"dim",dims[1],"nexp_",n_exp,".pdf"))
    # savefig(string("n_rejections_bydelta_corr",correlations[1],"dim",dims[1],"nexp_",n_exp,".pdf"))

else  # Plot as function of correlations
    n_rej_zzs_percentage = n_rej[1, :, :, 1, 1] / N
    mean_nrej_zzs_percent = transpose(mean(n_rej_zzs_percentage; dims = 1))
    n_rej_bps_percent = n_rej[2, :, :, 1, 1] / N
    mean_nrej_bps_percent = transpose(mean(n_rej_bps_percent; dims = 1))
    # n_rej_zzsgauss_percent = n_rej[3,:,:,1,1]/N
    # mean_nrej_zzsgauss_percent = transpose(mean(n_rej_zzsgauss_percent;dims=1))
    plot(
        correlations,
        mean_nrej_zzs_percent,
        ylabel = "Fraction of rejections",
        # xlabel = "Correlation",
        xlabel = "Variance of first component",
        linewidth = 2,
        legend = false,
        # legend=:topleft, legendfontsize=10,
        # xticks = [0, 0.25, 0.5, 0.75, 0.99],
        # ylim = [0,0.135],
        linecolor = colours[1],
        label = "ZZS - Splitting DBD",
    )
    plot!(
        correlations,
        mean_nrej_bps_percent,
        linewidth = 2,
        linecolor = colours[2],
        label = "BPS - Splitting RDBDR",
    )
    # savefig(string("n_rejections_bycorr_stepsize_",δ[1],"dim_",dims[1],"_nexp_",n_exp,".pdf"))
    # savefig(string("n_rejections_byvar_nolegend_stepsize_",δ[1],"dim_",dims[1],"_nexp_",n_exp,".pdf"))
end
