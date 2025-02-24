using Plots, ProgressBars
# using JLD2  # For saving data
include(joinpath(@__DIR__, "..", "helper_split.jl"))
include(joinpath(@__DIR__, "..","moves_samplers.jl"))
include(joinpath(@__DIR__, "..","algorithms.jl"))
include(joinpath(@__DIR__, "..","functions_particles.jl"))
include(joinpath(@__DIR__, "..","run_experiments.jl"))


N = 100 # number of particles

# To read the data
using DataFrames, CSV
df1 = CSV.read("particles_N=100_refrrate_1.0_K=5_M=3-gradevals-10000000.csv", DataFrame)
(errs,dels,nams) = read_data_give_errors_and_stepsizes(df1)
deltas_zigzag = dels[1]
deltas_bps = dels[2]
deltas_HMC_laplace = dels[3]
errors_zigzag = errs[1]
errors_bps = errs[2]
errors_HMC_laplace = errs[3]

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
delta = 5e-2
refreshment_rate = 1.
tolerance = 1e-8
iter_tolerance = 1 * 10^5
max_iters = 1 * 10^9
burnin = 1 * 10^8
cond_init = initial_position_particles(N,1.), initial_velocity_sphere(N)
func(z) = fun_var(z[1])
sampler_longrun(z) = bps_metropolised_energy_1step(interaction_pot,interaction_pot_grad,delta,z[1],z[2],refreshment_rate;v_sphere = false)
# estimates = long_run(sampler_longrun,initial_state_BPS,tolerance,iter_tolerance,max_iters,func; burn_in = burnin)
# println("Estimated value is $estimates")
# estimate = estimates
# estimate = (0.26966535488081433, 0.0545876783405469) # for N = 4, with HMC
# estimate = ((0.2658903572073838, 0.05294259036466889)) # for N = 4, with BPS
# estimate = (2.361880157178683, 2.1892957120182475) # for N = 10, with BPS
# estimate = (49.47722970322544, 45.01473279327602) # for N = 25, with BPS rel tol e-7
# estimate = (2328.9060416376865, 2737.2679824910224) # for N=100, with BPS 16hrs compute time
##
n_exp = 10
n_gradevals = 1 * 10^4     
nr_deltas = 16


## ZZS
ub = 1.
# deltas_zigzag = 10 .^ range(log10(1e-3), log10(3e-1), length = nr_deltas)
# Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j)
# ZZS_splitting_particles(delta,cond_init) = splitting_zzs_particles(gradV,Wrates,fun_var,ub,delta,N,n_gradevals,cond_init[1],cond_init[2])
# errors_zigzag = run_by_delta(ZZS_splitting_particles, initial_state_ZZ, deltas_zigzag, n_exp, estimate)
pl_mean = plot(deltas_zigzag,mean(errors_zigzag[1]; dims = 1)',xaxis=:log,yaxis=:log, label="ZZP",  lw = 2,linecolor=:blue)
pl_var  = plot(deltas_zigzag,mean(errors_zigzag[2]; dims = 1)',xaxis=:log,yaxis=:log, label="ZZP",  lw = 2,linecolor=:blue)

## BPS
ub_BPS = 1
refreshment_rate = 1. 
# deltas_bps = 10 .^ range(log10(1e-3), log10(7e-2), length=16)
# BPS_particles(delta,cond_init) = splitting_bps_particles(gradV,gradW,fun_var,ub_BPS,delta,N,n_gradevals,cond_init[1],cond_init[2],refreshment_rate)
# errors_bps = run_by_delta(BPS_particles, initial_state_BPS, deltas_bps, n_exp, estimate)
plot!(pl_mean,deltas_bps,transpose(mean(errors_bps[1]; dims = 1)),xaxis=:log,yaxis=:log, label="BPS",  lw = 2,linecolor=:green)
plot!(pl_var,deltas_bps,transpose(mean(errors_bps[2]; dims = 1)),xaxis=:log,yaxis=:log, label="BPS",  lw = 2,linecolor=:green)


## HMC with Laplace moments
# for N = 4
# K = 3
# M = 1
# for N = 25
K = 5
M = 3
# deltas_HMC_laplace = 10 .^ range(log10(5e-3), log10(6e-1), length = nr_deltas)
# n_iter_HMC = Int(ceil(n_gradevals / (M * (K + 2))))
# HMC_particles(delta,cond_init) = SHMC_Laplace(gradV,gradW,fun_var,N,delta,K,M,n_iter_HMC,cond_init[1],cond_init[2])
# errors_HMC_laplace = run_by_delta(HMC_particles, initial_state_HMC_laplace, deltas_HMC_laplace, n_exp,estimate)
plot!(pl_mean,deltas_HMC_laplace,mean(errors_HMC_laplace[1]; dims =1)',xaxis=:log,label="HMC", lw = 2,linecolor=:red)
plot!(pl_var, deltas_HMC_laplace,mean(errors_HMC_laplace[2]; dims =1)',xaxis=:log,label="HMC", lw = 2,linecolor=:red)


plot!(pl_mean,legend=:topleft,legendfontsize=12, ylabel = "Relative MSE", xlabel ="Step size",
        xguidefontsize=13,yguidefontsize=13)
plot!(pl_var,legend=:none, ylabel = "Relative MSE", xlabel ="Step size",xguidefontsize=13,yguidefontsize=13)
plot(
    pl_mean,  pl_var,
    layout = (1, 2),  # 1 row, 2 columns
    size = (600, 400) # Optional: Set figure size
)
# savefig(pl_mean,string("particles_stochgrad_mean_N=",N,"_sigma=",sigma,"n_grads=",n_gradevals,"_nexp=",n_exp,".pdf"))
# savefig(pl_var,string("particles_stochgrad_var_N=",N,"_sigma=",sigma,"n_grads=",n_gradevals,"_nexp=",n_exp,".pdf"))


## extra-grid search for HMC for parameters K,M
# n_exp = 50
# deltas_HMC_laplace = 10 .^ range(log10(1e-3), log10(6e-1), length = nr_deltas)
# Ks = [1,4,7,10]
# Ms = [1,4,7,10]
# vals = collect(Iterators.product(deltas_HMC_laplace,Ks,Ms))
# n_iter(v) = Int(ceil(n_gradevals / (v[3] * (v[2] + 2))))
# HMC_grid(val,cond_init) = SHMC_Laplace(gradV,gradW,fun_var,N,val[1],val[2],val[3],n_iter(val),cond_init[1],cond_init[2])
# errs_grid_HMC = grid_search(HMC_grid,initial_state_HMC_laplace,vals,n_exp,estimate)
# pl_grid_hmc = plot_errors_grid_HMC(errs_grid_HMC,vals)

# extra-grid search for BPS for the refreshment rate
# ub_BPS = 1
# deltas_BPS = 10 .^ range(log10(6e-3), log10(6e-1), length = nr_deltas)
# refs = [0.001,0.01,0.1,1.,10.]
# vals = collect(Iterators.product(deltas_BPS,refs))
# n_gradevals = 10^6
# BPS_grid(val,cond_init) = splitting_bps_particles(gradV,gradW,fun_var,ub_BPS,val[1],N,n_gradevals,cond_init[1],cond_init[2],val[2])
# errs_grid_BPS = grid_search(BPS_grid,initial_state_BPS,vals,n_exp,estimate)
# pl_grid_bps = plot_errors_grid_BPS(errs_grid_BPS,vals)

## MALTA
# D = sqrt(N)
# # deltas_MALTA = 10 .^ range(log10(7e-3), log10(6e-0), length = nr_deltas)
# deltas_MALTA = 10 .^ range(log10(7e-9), log10(6e-1), length = nr_deltas)
# iters_SMALTA = Int(ceil(n_gradevals / 2))  # each iterations of SMALTA has order 2d computations for the gradient
# MALTA_particles(delta,cond_init) = SMALTA_func(interaction_pot_grad_stoch,fun_var,delta,N,D,iters_SMALTA,cond_init)
# errors_MALTA = run_by_delta(MALTA_particles, initial_position, deltas_MALTA, n_exp,estimate)
# plot!(pl_mean,deltas_MALTA,mean(errors_MALTA[1]; dims =1)',label="MALTA", lw = 2,linecolor=:orange)
# plot!(pl_var,deltas_MALTA,mean(errors_MALTA[2]; dims =1)',label="MALTA", lw = 2,linecolor=:orange)


## To save the data
# using DataFrames, StatsPlots, CSV
# names_samplers = Vector{String}()
# push!(names_samplers,"ZZP")
# push!(names_samplers,"BPS")
# push!(names_samplers,"HMC")

# df = DataFrame(
#     sampler = String[],
#     step_size = Float64[],
#     error_mean = Float64[],
#     error_variance = Float64[]
# );
# total_err = (errors_zigzag,errors_bps,errors_HMC_laplace)
# total_deltas = (deltas_zigzag,deltas_bps,deltas_HMC_laplace)
# add_entries_errors!(df,total_err,names_samplers,total_deltas)
# CSV.write(string("particles_N=",N,"_refrrate_",refreshment_rate,"_K=", K,"_M=",M,"-gradevals-",n_gradevals,".csv"), df)
