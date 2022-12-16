using Plots
# using JLD2  # For saving data
using AdvancedHMC, Distributions, Plots, ForwardDiff
using LinearAlgebra
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("functions_particles.jl")

N = 10
# l = 100
# initial_θ = (collect(1:N) .- N / 2) ./ l
initial_θ = sort(randn(N))

# Define the target distribution
a = 1.0
V(r) = (1 / r^12) - (1 / r^6)
W(r) = a * sqrt(1 + r^2)

Vprime(r) = -12 * (1 / r^13) + 6 * (1 / r^7)
Wprime(r) = a * r / sqrt(1 + r^2)

interaction_pot(x)=sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))

function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end

ℓπ(θ) = -interaction_pot(θ)
#ℓπ(θ) = logpdf(MvNormal(zeros(N), I), θ)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 10000, 1000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(N)
#diff_fun(x) = [interaction_pot(x), interaction_pot_grad(x)]
hamiltonian = Hamiltonian(metric, ℓπ,  ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)


thin_factor = 1
thinned_samples = [samples[thin_factor*i+1] for i = 0 : convert(Int64,floor(n_samples/thin_factor))-1]
# p_aHMC = plot(reduce(hcat,samples)',legend=:no, title="Trajectories aHMC")
# display(p_aHMC)
p_aHMC = plot(reduce(hcat,thinned_samples)',
                legend=:no, 
                # ylims = [-20,20],
                title="Trajectories aHMC")
display(p_aHMC)
# savefig(p_aHMC, string("trajectories_advancedHMC.pdf"))

test_fun(x)=mean((x.-mean(x)).^2)
long_var_trace=Array{Float64}(undef, length(thinned_samples))
for iii=1:length(thinned_samples)
    long_var_trace[iii]=test_fun(thinned_samples[iii])
end
plot(long_var_trace, label = "AdvancedHMC", ylabel = "Empirical variance", xlabel = "Thinned iterations")
