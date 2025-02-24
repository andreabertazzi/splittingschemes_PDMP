using Plots, ProgressBars
# using JLD2  # For saving data
include(joinpath(@__DIR__, "..", "helper_split.jl"))
include(joinpath(@__DIR__, "..","moves_samplers.jl"))
include(joinpath(@__DIR__, "..","algorithms.jl"))
include(joinpath(@__DIR__, "..","functions_particles.jl"))
include(joinpath(@__DIR__, "..","run_experiments.jl"))


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
n_exp = 5
n_iterations = 1 * 10^8     
n_updates = 10^5
n_obs = Int(ceil(n_iterations/n_updates))

## ZZS
ub = 1.
deltas_zigzag = [8e-3,1e-2,3e-2]
Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j)
ZZS_splitting_particles(delta,cond_init) = splitting_zzs_particles(gradV,Wrates,fun_var,ub,delta,N,n_iterations,n_updates,cond_init[1],cond_init[2])
(errors_zigzag,runtimes_zzs) = grid_search_runtimes(ZZS_splitting_particles,initial_state_ZZ,deltas_zigzag,n_exp,n_obs,estimate)
runtimes_zzs_pl = hcat([vec(mean(hcat(runtimes_zzs...);dims=1) * i/n_obs) for i in 1:n_obs]...)'
mean_errors_zzs = hcat([vec(mean(errors_zigzag[i][1]; dims = 1)) for i in eachindex(errors_zigzag)]...)
var_errors_zzs = hcat([vec(mean(errors_zigzag[i][2]; dims = 1)) for i in eachindex(errors_zigzag)]...)
labels = hcat(["ZZP del = $val" for (_,val) in enumerate(deltas_zigzag)]...)
pl_mean_zz = plot(runtimes_zzs_pl,mean_errors_zzs, label=labels, yaxis=:log,  lw = 2,legendfontsize=4,)
pl_var_zz  = plot(runtimes_zzs_pl,var_errors_zzs, label=labels, yaxis=:log, lw = 2, legend = :none)
plot(
    pl_mean_zz,  pl_var_zz,
    layout = (1, 2),  # 1 row, 2 columns
    size = (600, 400), # Optional: Set figure size
    title = "ZZS"
)

## BPS
ub_BPS = 1
refreshment_rate = 1. 
deltas_bps = [3e-3, 5e-3, 7e-3]
BPS_particles(delta,cond_init) = splitting_bps_particles(gradV,gradW,fun_var,ub_BPS,delta,N,n_iterations,n_updates,cond_init[1],cond_init[2],refreshment_rate)
(errors_bps,runtimes_bps) = grid_search_runtimes(BPS_particles,initial_state_BPS,deltas_bps,n_exp,n_obs,estimate)
runtimes_bps_pl = hcat([vec(mean(hcat(runtimes_bps...);dims=1) * i/n_obs) for i in 1:n_obs]...)'
mean_errors_bps = hcat([vec(mean(errors_bps[i][1]; dims = 1)) for i in eachindex(errors_bps)]...)
var_errors_bps = hcat([vec(mean(errors_bps[i][2]; dims = 1)) for i in eachindex(errors_bps)]...)
labels_bps = hcat(["BPS del = $val" for (_,val) in enumerate(deltas_bps)]...)
pl_mean_bps = plot(runtimes_bps_pl,mean_errors_bps, label=labels_bps, yaxis=:log,  lw = 2,legendfontsize=4,)
pl_var_bps  = plot(runtimes_bps_pl,var_errors_bps, label=labels_bps, yaxis=:log, lw = 2, legend = :none)
plot(
    pl_mean_bps,  pl_var_bps,
    layout = (1, 2),  # 1 row, 2 columns
    size = (600, 400), # Optional: Set figure size
    title = "BPS"
)

## HMC with Laplace moments
# for N = 4
# K = 3
# M = 1
# for N = 25
K = 5
M = 3
deltas_HMC_laplace = [8e-3, 3e-2,5e-2,7e-2]
n_iterations_HMC = Int(ceil(n_iterations / 10))
n_obs_HMC = Int(ceil(n_iterations_HMC / n_updates))
HMC_particles(delta,cond_init) = SHMC_Laplace(gradV,gradW,fun_var,N,delta,K,M,n_iterations_HMC,n_updates,cond_init[1],cond_init[2])
(errors_HMC_laplace,runtimes_hmc) = grid_search_runtimes(HMC_particles,initial_state_HMC_laplace,deltas_HMC_laplace,n_exp,n_obs_HMC,estimate)
runtimes_hmc_pl = hcat([vec(mean(hcat(runtimes_hmc...);dims=1) * i/n_obs_HMC) for i in 1:n_obs_HMC]...)'
mean_errors_hmc = hcat([vec(mean(errors_HMC_laplace[i][1]; dims = 1)) for i in eachindex(errors_HMC_laplace)]...)
var_errors_hmc = hcat([vec(mean(errors_HMC_laplace[i][2]; dims = 1)) for i in eachindex(errors_HMC_laplace)]...)
labels_hmc = hcat(["HMC del = $val" for (_,val) in enumerate(deltas_HMC_laplace)]...)
pl_mean_hmc = plot(runtimes_hmc_pl,mean_errors_hmc, label=labels_hmc, yaxis=:log,  lw = 2, legendfontsize=4,)
pl_var_hmc  = plot(runtimes_hmc_pl,var_errors_hmc, label=labels_hmc, yaxis=:log, lw = 2, legend = :none)
plot(
    pl_mean_hmc,  pl_var_hmc,
    layout = (1, 2),  # 1 row, 2 columns
    size = (600, 400), # Optional: Set figure size
)


plot(
    pl_mean_zz,  pl_var_zz,
    pl_mean_bps,  pl_var_bps,
    pl_mean_hmc,  pl_var_hmc,
    layout = (3, 2),  # 1 row, 2 columns
    size = (600, 400), # Optional: Set figure size
)

# savefig(string("particles_byruntime_N=",N,"_refrrate_",refreshment_rate,"_K=", K,"_M=",M,".pdf"))