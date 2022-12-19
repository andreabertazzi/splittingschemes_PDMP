using Plots, BenchmarkTools, LinearAlgebra
using AdvancedHMC, Distributions, ForwardDiff
# using JLD2  # For saving data
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("functions_particles.jl")

N = 2 # number of particles
iter = 1 * 10^2 # number of iterations per thinned sample
thin_iter = 1 * 10^5 # number of thinned samples want to get. If ==1 then no thinning.
iter_UKLA = 10^6
thin_iter_UKLA = 5 * 10^4
δ = 1e-3
δ_UKLA = 1e-4
n_samples, n_adapts = 100000, 1000
run_ZZS = true
run_advHMC = true
run_UKLA = false


## Set up the target
a = 1.
V(r) = (1/r^12) - (1/r^6)
W(r) = a * sqrt(1 + r^2)

Vprime(r) = -12 * (1/r^13) + 6 * (1/r^7)
Wprime(r) = a * r / sqrt(1+r^2)

interaction_pot(x)=sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end

Vrates(x,v,i)   = define_Vrates(Vprime, x, v, i, N)
Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j, N)

l = 1
x_init = randn(N)/l
sort!(x_init)
v_init = rand((-1,1),N)

fun_var(x)=mean((x.-mean(x)).^2)
fun(x) = x.-mean(x)
p_var = plot()


## Run Zig-Zag 
if run_ZZS
    runtime_ZZS = @elapsed (chain_ZZS = splitting_zzs_particles(Vrates,Wrates,a,δ,N,iter,thin_iter,x_init,v_init));
    pos = getPosition(chain_ZZS)
    pl = plot(reduce(hcat,fun.(pos))',
        legend=:no, 
        title = "Trajectories uZZS",
#  xlims = [0,100],
    )
    display(pl)
    emp_var=Array{Float64}(undef, length(chain_ZZS))
    emp_var[1]=fun_var(pos[1])
    for iii=2:length(chain_ZZS)
        emp_var[iii]=emp_var[iii-1]+(fun_var(pos[iii])-emp_var[iii-1])/iii
    end

    p_var = plot!(p_var,emp_var, label = "ZZS", ylabel = "Empirical variance", xlabel = "Iterations")

    println("Run time for ZZS is $runtime_ZZS seconds")
end


## Run Advanced HMC

if run_advHMC
    initial_θ = x_init
    ℓπ(θ) = -interaction_pot(θ)

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(N)
    #diff_fun(x) = [interaction_pot(x), interaction_pot_grad(x)]
    hamiltonian = Hamiltonian(metric, ℓπ,  ForwardDiff)

    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    samples = Vector{Vector{Float64}}(undef,n_samples)
    runtime_advHMC = @elapsed((samples, stats) = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true));
    # samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
    pl_HMC = plot(reduce(hcat,fun.(samples))',
        legend=:no, 
        title = "Trajectories HMC",
        #  xlims = [0,100],
        )
    display(pl_HMC)

    var_advHMC=Array{Float64}(undef, length(samples))
    var_advHMC[1]=fun_var(samples[1])
    for iii=2:length(samples)
        var_advHMC[iii]=var_advHMC[iii-1]+(fun_var(samples[iii])-var_advHMC[iii-1])/iii
    end
    # p_var = plot!(long_var_av, label="HMC")

    pl_HMC = plot(reduce(hcat,fun.(samples))',
        legend=:no, 
        title = "Trajectories HMC",
        #  xlims = [0,100],
        )
    p_var = plot!(p_var,var_advHMC, label="HMC")

    println("Run time for AdvancedHMC is $runtime_advHMC seconds")

end

## Run UKLA
if run_UKLA
    ℓπ(θ) = -interaction_pot(θ)
    metric = DiagEuclideanMetric(N)
    hamiltonian = Hamiltonian(metric, ℓπ,  ForwardDiff)
    # δ = find_good_stepsize(hamiltonian, initial_θ)
    # δ = 0.008 
    iter = n_samples
    fric = 1.0;
    η = exp(-δ * fric)
    K = 1
    v_init=randn(size(x_init))

    runtime_UKLA = @elapsed(MD_UKLA = UKLA(interaction_pot_grad, δ_UKLA, iter_UKLA, K, η, x_init, v_init,N_store=thin_iter_UKLA))
    pos = getPosition(MD_UKLA)
    pl_UKLA = plot(reduce(hcat,fun.(pos))',
        legend=:no, 
        title = "Trajectories UKLA",
    )
    display(pl_UKLA)
    emp_var_UKLA=Array{Float64}(undef, length(MD_UKLA))
    emp_var_UKLA[1]=fun_var(pos[1])
    for iii=2:length(MD_UKLA)
        emp_var_UKLA[iii]=emp_var_UKLA[iii-1]+(fun_var(pos[iii])-emp_var_UKLA[iii-1])/iii
    end

    p_var = plot!(p_var,emp_var_UKLA, label = "UKLA", ylabel = "Empirical variance", xlabel = "Iterations")

    println("Run time for UKLA is $runtime_UKLA seconds")
    
end

# display(p_var)
## Plots variance
plot(p_var, ylims=[0,5])
# p_var = plot(emp_var, label = "ZZS", ylabel = "Empirical variance", xlabel = "Iterations")
# p_var = plot!(p_var,var_advHMC, label="HMC")
# plot!(var_advHMC, label="HMC")

# Trajectories 

# pos = getPosition(chain_ZZS; want_array=true)
# mu = mean(pos; dims = 1)

# savefig(pl, string("trajectories_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))
# p_bar = plot(mu', label="baricentre")
# savefig(p_bar, string("baricentre_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))
# trace_potential = [interaction_pot(pos[:,i]) for i=1:length(chain_ZZS)]
# p_tracepot = plot(trace_potential, label="Trace potential", yaxis=:log)
# savefig(p_tracepot, string("tracepot_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))

