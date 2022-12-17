using Plots, BenchmarkTools, LinearAlgebra
using AdvancedHMC, Distributions, ForwardDiff
# using JLD2  # For saving data
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("functions_particles.jl")

N = 30 # number of particles
iter = 1 * 10^3 # number of iterations per thinned sample
thin_iter = 1 * 10^4 # number of thinned samples want to get. If ==1 then no thinning.
δ = 1e-3
n_samples, n_adapts = 30000, 1000

a = 1.
V(r) = (1/r^12) - (1/r^6)
W(r) = a * sqrt(1 + r^2)

Vprime(r) = -12 * (1/r^13) + 6 * (1/r^7)
Wprime(r) = a * r / sqrt(1+r^2)

function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
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
    p_var = plot!(emp_var, label = "ZZS", ylabel = "Empirical variance", xlabel = "Iterations")

end


## Run Advanced HMC

if run_advHMC
    ℓπ(θ) = -interaction_pot(θ)

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(N)
    #diff_fun(x) = [interaction_pot(x), interaction_pot_grad(x)]
    hamiltonian = Hamiltonian(metric, ℓπ,  ForwardDiff)

    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    runtime_advHMC = @elapsed (samples, stats = sample(hamiltonian, proposal, x_init, n_samples, adaptor, n_adapts; progress=true))

    pl_HMC = plot(reduce(hcat,fun.(samples))',
        legend=:no, 
        title = "Trajectories HMC",
        #  xlims = [0,100],
        )
    display(pl_HMC)
    long_var_av=Array{Float64}(undef, length(samples))
    long_var_av[1]=fun_var(samples[1])
    for iii=2:length(samples)
        long_var_av[iii]=long_var_av[iii-1]+(fun_var(samples[iii])-long_var_av[iii-1])/iii
    end
    p_var = plot!(long_var_av, label="HMC")

end

display(p_var)



## Plots 

# Trajectories 

# pos = getPosition(chain_ZZS; want_array=true)
# mu = mean(pos; dims = 1)

# savefig(pl, string("trajectories_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))
# p_bar = plot(mu', label="baricentre")
# savefig(p_bar, string("baricentre_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))
# trace_potential = [interaction_pot(pos[:,i]) for i=1:length(chain_ZZS)]
# p_tracepot = plot(trace_potential, label="Trace potential", yaxis=:log)
# savefig(p_tracepot, string("tracepot_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))

