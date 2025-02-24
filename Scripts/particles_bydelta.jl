using Plots
# using JLD2  # For saving data
include(joinpath(@__DIR__, "..", "helper_split.jl"))
include(joinpath(@__DIR__, "..","moves_samplers.jl"))
include(joinpath(@__DIR__, "..","algorithms.jl"))
include(joinpath(@__DIR__, "..","functions_particles.jl"))
include(joinpath(@__DIR__, "..","run_experiments.jl"))


N = 4 # number of particles

# Choose potential functions V and W and their derivatives
V(r) = r^4
W(r) = - sqrt(1 + r^2)
Vprime(r) = 4 * (r^3)
Wprime(r) = r / W(r)

# Define potential and its gradient
function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end
function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
end
# function target_unnorm(x::Vector{Float64})
#     return exp(-interaction_pot(x))
# end

# Define the test function 
fun_var(x)=mean((x.-mean(x)).^2)

## Initial conditions
sigma = 1.
initial_state_ZZ() = initial_position_particles(N,sigma), rand((-1,1),N)
initial_position() = initial_position_particles(N,sigma)
initial_state_BPS() = initial_position_particles(N,sigma), randn(N)
initial_state_HMC_laplace() = initial_position_particles(N,sigma), draw_laplace(N)

## One long, adjusted run to estimate the true value
# delta = 1e-1
# refreshment_rate = 1.
# tolerance = 1e-7
# iter_tolerance = 2 * 10^5
# max_iters = 1 * 10^8
# cond_init = initial_position_particles(N,1.), initial_velocity_sphere(N)
# func(z) = fun_var(z[1])
# sampler_longrun(z) = bps_metropolised_energy_1step(interaction_pot,interaction_pot_grad,delta,z[1],z[2],refreshment_rate;v_sphere = false)
# estimates = long_run(sampler_longrun,initial_state_BPS,tolerance,iter_tolerance,max_iters,func)
# println("Estimated value is $estimate")
estimate = (0.26966535488081433, 0.0545876783405469) # for N = 4, new, with HMC
# estimate = 0.26573 # for N = 4, new, with ZZS

##
n_exp = 500
n_iterations = 1 * 10^4    # Number of gradient evaluations used for each experiment. 
nr_deltas = 8

## ZZS
# deltas_zigzag = 10 .^ range(log10(6e-3), log10(3.5e-1), length=16)
deltas_zigzag = 10 .^ range(log10(7e-3), log10(6e-0), length = nr_deltas)
# deltas_zigzag = 10 .^ range(log10(1e-2), log10(6e-1), length=16)
ZZS_splitting_particles(delta,cond_init) = zzs_metropolised_energy(interaction_pot,interaction_pot_grad,fun_var,delta,n_iterations,cond_init[1],cond_init[2])
# errors_zigzag = run_by_delta(ZZS_splitting_particles, initial_state_ZZ, deltas_zigzag, n_exp, estimate)
pl_mean = plot(deltas_zigzag,mean(errors_zigzag[1]; dims = 1)',xaxis=:log,yaxis=:log, label="ZZP",  lw = 2,linecolor=:blue)
pl_var  = plot(deltas_zigzag,mean(errors_zigzag[2]; dims = 1)',xaxis=:log,yaxis=:log, label="ZZP",  lw = 2,linecolor=:blue)

## BPS
refreshment_rate = 1.
# deltas_bps = 10 .^ range(log10(1e-2), log10(4e-0), length=16)
deltas_bps = 10 .^ range(log10(6e-2), log10(14e0), length = nr_deltas)
BPS_particles(delta,cond_init) = bps_metropolised_energy(interaction_pot,interaction_pot_grad,fun_var,delta,n_iterations,cond_init[1],cond_init[2],refreshment_rate)
# errors_bps = run_by_delta(BPS_particles, initial_state_BPS, deltas_bps, n_exp, estimate)
plot!(pl_mean,deltas_bps,mean(errors_bps[1]; dims = 1)',xaxis=:log,yaxis=:log, label="BPS",  lw = 2,linecolor=:green)
plot!(pl_var,deltas_bps,mean(errors_bps[2]; dims = 1)',xaxis=:log,yaxis=:log, label="BPS",  lw = 2,linecolor=:green)

## MALTA
D = sqrt(N)
MALTA_particles(delta,cond_init) = MALTA(interaction_pot,interaction_pot_grad,fun_var,delta,D,n_iterations,cond_init)
# deltas_MALTA = 10 .^ range(log10(1e-0), log10(10), length=16)
deltas_MALTA = 10 .^ range(log10(5e-3),log10(3e0), length = nr_deltas)
# errors_MALTA = run_by_delta(MALTA_particles, initial_position, deltas_MALTA, n_exp,estimate)
plot!(pl_mean,deltas_MALTA,mean(errors_MALTA[1]; dims =1)',label="MALTA", lw = 2,linecolor=:orange)
plot!(pl_var,deltas_MALTA,mean(errors_MALTA[2]; dims =1)',label="MALTA", lw = 2,linecolor=:orange)

## HMC with Laplace moments
# deltas_HMC_laplace = 10 .^ range(log10(1e-2), log10(8e-1), length=16)
deltas_HMC_laplace = 10 .^ range(log10(9.5e-3), log10(4e-1), length = nr_deltas)
K = 8
n_iterations_HMC = Int(ceil(n_iterations * 2 / (K+1)))
HMC_particles(delta,cond_init) = HMC_Laplace_RS(interaction_pot,interaction_pot_grad,fun_var,delta,K,n_iterations_HMC,cond_init[1],cond_init[2])
errors_HMC_laplace = run_by_delta(HMC_particles, initial_state_HMC_laplace, deltas_HMC_laplace, n_exp,estimate)
plot!(pl_mean,deltas_HMC_laplace,mean(errors_HMC_laplace[1]; dims =1)',xaxis=:log,label="HMC", lw = 2,linecolor=:red)
plot!(pl_var, deltas_HMC_laplace,mean(errors_HMC_laplace[2]; dims =1)',xaxis=:log,label="HMC", lw = 2,linecolor=:red)

plot!(pl_mean,legend=:topright, title = "rel-MSE for mean")
plot!(pl_var,legend=:topright, title = "rel-MSE for variance")
plot(
    pl_mean,  pl_var,
    layout = (1, 2),  # 1 row, 2 columns
    size = (600, 400) # Optional: Set figure size
)

# savefig(string("particles_N=",N,"_sigma=",sigma,"n_iter=",thin_iter,"_nexp=",n_exp,".pdf"))


## extra-grid search for HMC
# deltas_HMC = [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1,]
# Ks = [3,6,10,15]
# n_iterations_HMC = [Int(ceil(n_iterations * 2 / (Ks[i] + 1))) for i in eachindex(Ks)]
# pl_m = plot(title="Mean")
# pl_v = plot(title="Var")
# for i in eachindex(Ks)
#     HMC_particles(delta,cond_init) = HMC_Laplace(interaction_pot,interaction_pot_grad,fun_var,delta,Ks[i],n_iterations_HMC[i],cond_init[1],cond_init[2])
#     errors_HMC_grid = run_by_delta(HMC_particles, initial_state_HMC_laplace, deltas_HMC, n_exp,estimate)
#     val = Ks[i]
#     plot!(pl_m,deltas_HMC,median(errors_HMC_grid[1]; dims =1)',xaxis=:log,yaxis=:log,label="HMC (K=$val)", lw=2)
#     plot!(pl_v,deltas_HMC,median(errors_HMC_grid[1]; dims =1)',xaxis=:log,yaxis=:log,label="HMC (K=$val)", lw=2)   
# end
# plot(
#     pl_m,  pl_v,
#     layout = (1, 2),  # 1 row, 2 columns
#     size = (600, 400) # Optional: Set figure size
# )

## extra-grid search for BPS
# lambdas = [0.1, 0.5, 1., 3.,8.,10.]
# deltas_bps = 10 .^ range(log10(1e-2), log10(10e-0), length=8)
# pl_m = plot(title="Mean")
# pl_v = plot(title="Var")
# for (i,lam) in enumerate(lambdas)
#     BPS_particles(delta,cond_init) = bps_metropolised_energy(interaction_pot,interaction_pot_grad,fun_var,delta,n_iterations,cond_init[1],cond_init[2],lambdas[i])
#     errs = run_by_delta(BPS_particles, initial_state_BPS, deltas_bps, n_exp, estimate)
#     plot!(pl_m,deltas_bps,mean(errs[1]; dims = 1)',xaxis=:log,yaxis=:log, label="lambda = $lam",  lw = 2)
#     plot!(pl_v,deltas_bps,mean(errs[2]; dims = 1)',xaxis=:log,yaxis=:log, label="lambda = $lam",  lw = 2)
# end
# plot(
#     pl_m,  pl_v,
#     layout = (1, 2),  # 1 row, 2 columns
#     size = (600, 400) # Optional: Set figure size
# )
