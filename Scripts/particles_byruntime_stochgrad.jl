using Plots, ProgressBars
# using JLD2  # For saving data
include(joinpath(@__DIR__, "..", "helper_split.jl"))
include(joinpath(@__DIR__, "..","moves_samplers.jl"))
include(joinpath(@__DIR__, "..","algorithms.jl"))
include(joinpath(@__DIR__, "..","functions_particles.jl"))
include(joinpath(@__DIR__, "..","run_experiments.jl"))

# To read the data
using DataFrames, CSV
df1 = CSV.read("particles_byruntime_N=25_refrrate_1.0_K=5_M=3.csv", DataFrame)
(errors,all_runtimes,names_samplers) = read_data_give_errors_and_runtimes(df1)
mean_errors_zzs = errors[1]
# mean_errors_bps = errors[2]
# mean_errors_hmc = errors[3]

runtimes_zzs = all_runtimes[1]
# runtimes_bps = all_runtimes[2]
# runtimes_hmc = all_runtimes[3]
names_samplers = unique(df1.sampler)
n_obs_zzs = nrow(df1[df1.sampler .== names_samplers[1],:])
# n_obs_bps = nrow(df1[df1.sampler .== names_samplers[2],:])

N = 25 # number of particles

# Choose potential functions V and W and their derivatives
V(r) = r^4
W(r) = - sqrt(1 + r^2)
Vprime(r) = 4 * (r^3)
Wprime(r) = r / W(r)

# Define potential and its gradient
function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end
function interaction_pot_grad_stoch(x::Vector{Float64},j::Integer) 
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + Wprime.(x .- x[j]) )
end
function gradV(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])))
end
function gradW(x::Vector{Float64},j::Integer)
    return vec(Wprime.(x .- x[j]))
end
function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
end


# Define the test function 
fun_var(x) = mean((x.-mean(x)).^2)

## Initial conditions
sigma = 1.
initial_state_ZZ() = initial_position_particles(N,sigma), rand((-1,1),N)
initial_position() = initial_position_particles(N,sigma)
initial_state_BPS() = initial_position_particles(N,sigma), randn(N)
initial_state_HMC_laplace() = initial_position_particles(N,sigma), draw_laplace(N)

## One long, adjusted run to estimate the true value
# delta = 5e-2
# refreshment_rate = 1.
# tolerance = 1e-8
# iter_tolerance = 1 * 10^5
# max_iters = 1 * 10^9
# burnin = 1 * 10^8
# cond_init = initial_position_particles(N,1.), initial_velocity_sphere(N)
# func(z) = fun_var(z[1])
# sampler_longrun(z) = bps_metropolised_energy_1step(interaction_pot,interaction_pot_grad,delta,z[1],z[2],refreshment_rate;v_sphere = false)
# estimates = long_run(sampler_longrun,initial_state_BPS,tolerance,iter_tolerance,max_iters,func; burn_in = burnin)
# println("Estimated value is $estimates")
# estimate = estimates
# estimate = ((0.2658903572073838, 0.05294259036466889)) # for N = 4, with BPS
# estimate = (2.361880157178683, 2.1892957120182475) # for N = 10, with BPS
estimate = (49.47722970322544, 45.01473279327602) # for N = 25, with BPS rel tol e-7
# estimate = (2328.9060416376865, 2737.2679824910224) # for N=100, with BPS 16hrs compute time

##
n_exp = 10
n_iterations = 1 * 10^8
n_updates = 1 * 10^5
# n_iterations = 1 * 10^9     
# n_updates = 1 * 10^6
n_obs = Int(ceil(n_iterations/n_updates))

## ZZS
ub = 1.
deltas_zigzag = 3e-2
# n_obs_zzs = n_obs
Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j)
ZZS_splitting_particles(delta,cond_init) = splitting_zzs_particles(gradV,Wrates,fun_var,ub,delta,N,n_iterations,n_updates,cond_init[1],cond_init[2])
# (errors_zigzag,runtimes_zzs) = run_by_iters(ZZS_splitting_particles, initial_state_ZZ, deltas_zigzag, n_exp, n_obs_zzs, estimate)
# mean_errors_zzs = (transpose(mean(errors_zigzag[1]; dims = 1)),transpose(mean(errors_zigzag[2]; dims = 1)))
runtimes_zzs_pl = [mean(runtimes_zzs) * i/n_obs_zzs for i in 1:n_obs_zzs]
pl_mean = plot(runtimes_zzs_pl,mean_errors_zzs[1], label="ZZS",  lw = 2,linecolor=:blue)
pl_var  = plot(runtimes_zzs_pl,mean_errors_zzs[2], label="ZZS",  lw = 2,linecolor=:blue)

## BPS
ub_BPS = 1
refreshment_rate = 1. 
deltas_bps = 4e-3
n_iterations_bps = Int(ceil(1.8 * n_iterations))
n_obs_bps = Int(ceil(n_iterations_bps / n_updates))
BPS_particles(delta,cond_init) = splitting_bps_particles(gradV,gradW,fun_var,ub_BPS,delta,N,n_iterations_bps,n_updates,cond_init[1],cond_init[2],refreshment_rate)
(errors_bps,runtimes_bps) = run_by_iters(BPS_particles, initial_state_BPS, deltas_bps, n_exp, n_obs_bps, estimate)
mean_errors_bps = (transpose(mean(errors_bps[1]; dims = 1)),transpose(mean(errors_bps[2]; dims = 1)))
runtimes_bps_pl = [mean(runtimes_bps) * i/n_obs_bps for i in 1:n_obs_bps]
plot!(pl_mean,runtimes_bps_pl,mean_errors_bps[1], label="BPS",  lw = 2,linecolor=:green)
plot!(pl_var,runtimes_bps_pl,mean_errors_bps[2], label="BPS",  lw = 2,linecolor=:green)


## HMC with Laplace moments
# for N = 4
# K = 3
# M = 1
# for N = 25
K = 5
M = 3
deltas_HMC_laplace = 5e-3
n_iterations_HMC = Int(ceil(n_iterations / 7))
n_obs_HMC = Int(ceil(n_iterations_HMC / n_updates))
HMC_particles(delta,cond_init) = SHMC_Laplace(gradV,gradW,fun_var,N,delta,K,M,n_iterations_HMC,n_updates,cond_init[1],cond_init[2])
(errors_HMC_laplace,runtimes_hmc) = run_by_iters(HMC_particles, initial_state_HMC_laplace, deltas_HMC_laplace, n_exp, n_obs_HMC, estimate)
mean_errors_hmc = (transpose(mean(errors_HMC_laplace[1]; dims = 1)),transpose(mean(errors_HMC_laplace[2]; dims = 1)))
runtimes_hmc_pl = [mean(runtimes_hmc) * i/n_obs_HMC for i in 1:n_obs_HMC]
plot!(pl_mean,runtimes_hmc_pl,mean_errors_hmc[1],label="HMC", lw = 2,linecolor=:red)
plot!(pl_var, runtimes_hmc_pl,mean_errors_hmc[2],label="HMC", lw = 2,linecolor=:red)


plot!(pl_mean, ylabel = "Relative MSE", xlabel ="Run time",
        legend=:topright,legendfontsize=12, xguidefontsize=13,yguidefontsize=13, 
        # xaxis=:log,
        xlims = [-.5,minimum([mean(runtimes_zzs),mean(runtimes_bps),mean(runtimes_hmc)])],
        # ylims = [1e-5,1e-2],
        yaxis=:log
        )

plot!(pl_var, ylabel = "Relative MSE", xlabel ="Run time",
        legend=:none,xguidefontsize=13,yguidefontsize=13,
        xlims = [-.5,minimum([mean(runtimes_zzs),mean(runtimes_bps),mean(runtimes_hmc)])],
        # ylims = [1e-4,1e3],
        # xaxis=:log,
        yaxis=:log
        )
plot(
    pl_mean,  pl_var,
    layout = (1, 2),  # 1 row, 2 columns
    size = (600, 400) # Optional: Set figure size
)
# savefig(pl_mean,string("particles_stochgrad_byruntime_mean_N=",N,"_sigma=",sigma,"n_iters=",n_iterations,"_nexp=",n_exp,".pdf"))
# savefig(pl_var,string("particles_stochgrad_byruntime_var_N=",N,"_sigma=",sigma,"n_iters=",n_iterations,"_nexp=",n_exp,".pdf"))

## To save the data
# using DataFrames, StatsPlots, CSV
# names_samplers = Vector{String}()
# push!(names_samplers,"ZZP del = $deltas_zigzag")
# push!(names_samplers,"BPS del = $deltas_bps")
# push!(names_samplers,"HMC del = $deltas_HMC_laplace")

# df = DataFrame(
#     sampler = String[],
#     runtime = Float64[],
#     error_mean = Float64[],
#     error_variance = Float64[]
# );
# total_mean_errors = (mean_errors_zzs,mean_errors_bps,mean_errors_hmc)
# mean_runtimes = [mean(runtimes_zzs),mean(runtimes_bps),mean(runtimes_hmc)]
# add_entries_errors_runtime!(df,total_mean_errors,names_samplers,mean_runtimes)
# CSV.write(string("particles_byruntime_N=",N,"_refrrate_",refreshment_rate,"_K=", K,"_M=",M,".csv"), df)

# To read the data
# df1 = CSV.read("particles_byruntime_N=100_refrrate_1.0_K=5_M=3.csv", DataFrame)
# (errors,all_runtimes,names_samplers) = read_data_give_errors_and_runtimes(df1)
# mean_errors_zzs = errors[1]
# mean_errors_bps = errors[2]
# mean_errors_hmc = errors[3]

# runtimes_hmc = all_runtimes[3]
# runtimes_zzs = all_runtimes[1]
# runtimes_bps = all_runtimes[2]