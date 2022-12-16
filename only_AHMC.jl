using Plots
# using JLD2  # For saving data
using AdvancedHMC, Distributions, Plots
using LinearAlgebra
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("functions_particles.jl")

N = 5
# l = 100
# initial_θ = (collect(1:N) .- N / 2) ./ l
initial_θ = sort(randn(N))

# Define the target distribution
a = 1.0
V(r) = (1 / r^12) - (1 / r^6)
W(r) = a * sqrt(1 + r^2)

Vprime(r) = -12 * (1 / r^13) + 6 * (1 / r^7)
Wprime(r) = a * r / sqrt(1 + r^2)

function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
end

function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end

ℓπ(θ) = -interaction_pot(θ)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 10_000_000, 200_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(N)
diff_fun(x) = [interaction_pot(x), interaction_pot_grad(x)]
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
# initial_ϵ = 1e-10
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw sampstatsles from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)

thin_factor = 100
thinned_samples = [samples[thin_factor*i+1] for i = 0 : convert(Int64,floor(n_samples/thin_factor))-1]
# p_aHMC = plot(reduce(hcat,samples)',legend=:no, title="Trajectories aHMC")
# display(p_aHMC)
p_aHMC = plot(reduce(hcat,thinned_samples)',legend=:no, title="Trajectories aHMC")
display(p_aHMC)
# savefig(p_aHMC, string("trajectories_advancedHMC.pdf"))

emp_var = compute_variance_particles(thinned_samples, vec(mu))
plot(emp_var, label = "ZZS", ylabel = "Empirical variance", xlabel = "Thinned iterations")
