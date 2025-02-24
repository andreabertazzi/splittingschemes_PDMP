include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("run_experiments.jl")

## If want to count the number of rejections and make nice plots, run this

# δ = [0.3]
δ = [0.05*i for i=1:20]
# N = Integer(round(T/δ))
T = 10^3
# N = 10^3
n_exp = 30
# n_exp = 500
refresh_rate = 0.5
# dims = [2,3,4,5,6,7,8,9,10,12,15,17,20,25,30,40,50,75,100]
dims = [20]
dims = [1, 5, 15, 20, 50, 80, 150, 250,500,750,1000,1250,1500,1750,2000,2500]
radius(x) = sum(x .^ 2)
test_func = radius

## For correlated covariance matrix
# correlations = [0.1*i for i=0:9]
# second_bit = [(0.9+0.02*i) for i=1:4]
# correlations = vcat(correlations,second_bit)
correlations = [0.5]
# correlations = [0.0, 0.15, 0.3, 0.45, 0.6, 0.9]
Cov(ρ,d) = ρ * ones(d, d) + (1 - ρ)I
Cov_inv(ρ,d) = inv(Cov(ρ,d))
# truth = compute_radius(Cov,correlations,dims)
# # truth = Float64(sum(diag(Σ)))

## For diagonal covariance matrix
# correlations = [0.1]
# correlations = [0.00001]
# correlations = [0.0001*(i) for i=1:40:1001]
# correlations = [0.0001,0.0005,0.001, 0.003, 0.007,0.0085, 0.01, 0.03, 0.08, 0.1, 0.15, 0.2]
# correlations = [0.0001,0.0005,0.001, 0.003, 0.007,0.01,0.02,0.03,0.04,0.05, 0.07,0.09,0.13,0.2]

# d_half(d) = Integer(ceil(d/2))
# Cov(a,d) = diagm([a*ones(d_half(d));ones(d-d_half(d))])
# Cov_inv(a,d) = diagm([1/a*ones(d_half(d));ones(d-d_half(d))])
# ρ = 0.5
# function Cov_wcorr(a,d,ρ)
#     A = ρ * ones(d, d) + (1 - ρ)*diagm(ones(d))
#     A[1,1] = a
#     A[2:end,1] .= ρ * sqrt(a)
#     A[1,2:end] .= ρ * sqrt(a)
#     A
# end
# Cov(a,d) = Cov_wcorr(a,d,.5)
# Cov_inv(a,d) = inv(Cov(a,d))
# Cov(a,d) = diagm([a;ones(d-1)])
# Cov_inv(a,d) = diagm([1/a;ones(d-1)])

truth = compute_radius(Cov,correlations,dims)
∇U(x, Σ_inv) = Σ_inv * x
target_unnorm(x, Σ_inv) = exp(-dot(x, Σ_inv * x) / 2)

sampler_metropolis = [zzs_metropolised,  bps_metropolised]
sampler_unadjusted = [splitting_zzs_DBD, splitting_bps_RDBDR_gauss]
sampler_continuous = [ZigZag, BPS]
draw_v             = [draw_velocity_zzs, draw_velocity_gauss]
refresh_rates      = [0.0, 0.5]

(err_discr, err_cts) = compute_errors_discrete_and_cts_gauss(
                            sampler_metropolis,
                            sampler_unadjusted,
                            sampler_continuous,
                            draw_v,
                            draw_v,
                            target_unnorm,
                            Cov,
                            Cov_inv,
                            ∇U,
                            δ,
                            dims,
                            correlations,
                            n_exp,
                            T,
                            test_func,
                            truth,
                            refresh_rates,
                            0.01
                            )


## PLOTS

colours = [:red, :blue, :green]

if length(dims) > 1
    mean_err_zzs_metropolis = transpose(mean(err_discr[1, :, 1, :, 1]; dims = 1))
    mean_err_bps_metropolis = transpose(mean(err_discr[2, :, 1, :, 1]; dims = 1))
    mean_err_zzs_unadj      = transpose(mean(err_discr[3, :, 1, :, 1]; dims = 1))
    mean_err_bps_unadj      = transpose(mean(err_discr[4, :, 1, :, 1]; dims = 1))
    mean_err_zzs_cts        = transpose(mean(err_cts[1, :, 1, :]; dims = 1))
    mean_err_bps_cts        = transpose(mean(err_cts[2, :, 1, :]; dims = 1))
    # mean_err_zzs_metropolis = transpose(mean(err_discr[1, :, 1, :, 1]; dims = 1))
    # mean_err_bps_metropolis = transpose(mean(err_discr[2, :, 1, :, 1]; dims = 1))
    # n_rej_zzsgauss_percent = n_rej[3,:,1,:,1]/N
    # mean_nrej_zzsgauss_percent = transpose(mean(n_rej_zzsgauss_percent;dims=1))
    p = plot(
        dims,
        mean_err_zzs_metropolis,
        ylabel = "Error radius",
        # xlabel = "Correlation",
        xlabel = "Dimension",
        linewidth = 2,
        # legend = false,
        legend=:topleft, legendfontsize=10,
        xticks = [1, 25, 50, 75, 100],
        # ylim = [0,8],
        linecolor = colours[1],
        label = "ZZS (adjusted)",
    )
    plot!(
        dims,
        mean_err_bps_metropolis,
        linewidth = 2,
        linecolor = colours[2],
        label = "BPS (adjusted)",
    )
    plot!(
        dims,
        mean_err_zzs_unadj,
        linewidth = 2,
        linecolor = colours[1],
        line =:dash,
        label = "ZZS (unadjusted)",
    )
    plot!(
        dims,
        mean_err_bps_unadj,
        linewidth = 2,
        line =:dash,
        linecolor = colours[2],
        label = "BPS (unadjusted)",
    )
    plot!(
        dims,
        mean_err_zzs_cts,
        linewidth = 2,
        linecolor = colours[1],
        line =:dot,
        label = "ZZS (continuous)",
    )
    plot!(
        dims,
        mean_err_bps_cts,
        linewidth = 2,
        line =:dot,
        linecolor = colours[2],
        label = "BPS (continuous)",
    )
    # savefig(string("error_bydims_delta_",δ[1],"_corr_",correlations[1],"_nexp_",n_exp,".pdf"))
    # savefig(string("error_bydims_delta_",δ[1],"_var_",correlations[1],"_nexp_",n_exp,".pdf"))
    # savefig(string("error_bydims_delta_",δ[1],"_var_",correlations[1],"_nexp_",n_exp,"_nolegend_.pdf"))
elseif length(δ) > 1 # Plot as function of step size
    mean_err_zzs_metropolis = transpose(mean(err_discr[1, :, 1, 1, :]; dims = 1))
    mean_err_bps_metropolis = transpose(mean(err_discr[2, :, 1, 1, :]; dims = 1))
    mean_err_zzs_unadj      = transpose(mean(err_discr[3, :, 1, 1, :]; dims = 1))
    mean_err_bps_unadj      = transpose(mean(err_discr[4, :, 1, 1, :]; dims = 1))
    mean_err_zzs_cts        = mean(err_cts[1, :, 1, 1]; dims = 1)
    mean_err_bps_cts        = mean(err_cts[2, :, 1, 1]; dims = 1)
    plot(
        δ,
        mean_err_zzs_metropolis,
        ylabel = "Error radius",
        xlabel = "Step size",
        # xlabel = "Variance of first component",
        linewidth = 2,
        # legend = false,
        legend=:topleft, legendfontsize=10,
        # xticks = [0, 0.25, 0.5, 0.75, 0.99],
        ylim = [0,6],
        linecolor = colours[1],
        label = "ZZS (adjusted)",
    )
    plot!(
        δ,
        mean_err_bps_metropolis,
        linewidth = 2,
        linecolor = colours[2],
        label = "BPS (adjusted)",
    )
    plot!(
        δ,
        mean_err_zzs_unadj,
        linewidth = 2,
        linecolor = colours[1],
        line =:dash,
        label = "ZZS (unadjusted)",
    )
    plot!(
        δ,
        mean_err_bps_unadj,
        linewidth = 2,
        line =:dash,
        linecolor = colours[2],
        label = "BPS (unadjusted)",
    )
    plot!(
        δ,
        mean_err_zzs_cts .* ones(length(δ)),
        linewidth = 2,
        linecolor = colours[1],
        line =:dot,
        label = "ZZS (continuous)",
    )
    plot!(
        δ,
        mean_err_bps_cts .* ones(length(δ)),
        linewidth = 2,
        line =:dot,
        linecolor = colours[2],
        label = "BPS (continuous)",
    )
    # savefig(string("error_bydelta_var",correlations[1],"dim",dims[1],"nexp_",n_exp,".pdf"))
    # savefig(string("error_bydelta_nolegend_var",correlations[1],"dim",dims[1],"nexp_",n_exp,".pdf"))
    # savefig(string("error_bydelta_corr",correlations[1],"dim",dims[1],"nexp_",n_exp,".pdf"))

else  # Plot as function of correlations
    mean_err_zzs_metropolis = transpose(mean(err_discr[1, :, :, 1, 1]; dims = 1))
    mean_err_bps_metropolis = transpose(mean(err_discr[2, :, :, 1, 1]; dims = 1))
    mean_err_zzs_unadj      = transpose(mean(err_discr[3, :, :, 1, 1]; dims = 1))
    mean_err_bps_unadj      = transpose(mean(err_discr[4, :, :, 1, 1]; dims = 1))
    mean_err_zzs_cts        = transpose(mean(err_cts[1, :, :, 1]; dims = 1))
    mean_err_bps_cts        = transpose(mean(err_cts[2, :, :, 1]; dims = 1))
    plot(
        correlations,
        mean_err_zzs_metropolis,
        ylabel = "Error radius",
        # xlabel = "Correlation",
        xlabel = "Variance of first component",
        linewidth = 2,
        legend = false,
        # legend=:topleft, legendfontsize=10,
        # xticks = [0, 0.25, 0.5, 0.75, 0.99],
        ylim = [0,10],
        linecolor = colours[1],
        label = "ZZS (adjusted)",
    )
    plot!(
        correlations,
        mean_err_bps_metropolis,
        linewidth = 2,
        linecolor = colours[2],
        label = "BPS (adjusted)",
    )
    plot!(
        correlations,
        mean_err_zzs_unadj,
        linewidth = 2,
        linecolor = colours[1],
        line =:dash,
        label = "ZZS (unadjusted)",
    )
    plot!(
        correlations,
        mean_err_bps_unadj,
        linewidth = 2,
        line =:dash,
        linecolor = colours[2],
        label = "BPS (unadjusted)",
    )
    plot!(
        correlations,
        mean_err_zzs_cts,
        linewidth = 2,
        linecolor = colours[1],
        line =:dot,
        label = "ZZS (continuous)",
    )
    plot!(
        correlations,
        mean_err_bps_cts,
        linewidth = 2,
        line =:dot,
        linecolor = colours[2],
        label = "BPS (continuous)",
    )
    # savefig(string("error_bycorr_stepsize_",δ[1],"dim_",dims[1],"_nexp_",n_exp,".pdf"))
    # savefig(string("error_byvar_stepsize_",δ[1],"dim_",dims[1],"_nexp_",n_exp,".pdf"))
    # savefig(string("error_byvar_nolegend_stepsize_",δ[1],"dim_",dims[1],"_nexp_",n_exp,".pdf"))
end
