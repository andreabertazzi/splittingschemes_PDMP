using AdvancedHMC, Distributions, Plots
using LinearAlgebra
include("algorithms.jl")
# Choose parameter dimensionality and initial parameter value
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

function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
end

function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end

ℓπ(θ) = -interaction_pot(θ)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 10_000, 2_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(N)
diff_fun(x) = [interaction_pot(x), interaction_pot_grad(x)]
hamiltonian = Hamiltonian(metric, ℓπ, diff_fun)

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

p_aHMC = plot(reduce(hcat,samples)',legend=:no, title="Trajectories aHMC")
display(p_aHMC)
# savefig(p_aHMC, string("trajectories_advancedHMC.pdf"))


## Running our own codes
x_init = initial_θ
# sort!(x_init)
v_init = randn(N)
# δ = initial_ϵ
δ = 1e-20
iter = 1 * 10^4
γ = 1.0;
η = exp(-δ * γ)
K = 1

x_init=initial_θ
v_init=randn(size(x_init))

MD_UKLA = UKLA(interaction_pot_grad, δ, iter, K, η, x_init, v_init)
MD_HMC = HMC(interaction_pot_grad, interaction_pot, δ, iter, K, η, x_init, v_init)
plot(reduce(hcat,getPosition(MD_UKLA))', ylims = [-5,5], xlims = [1,500],legend=:no, title="Trajectories UKLA" )
p = plot(reduce(hcat,getPosition(MD_HMC))', ylims = [-5,5], xlims = [1,5000],legend=:no, title="Trajectories HMC" )

# savefig(p, string("trajectories_HMC.pdf"))

#display(interaction_pot(MD_UKLA[1].position))
logpi_trace_UKLA = vec(getPosition(MD_UKLA, want_array=true, observable=interaction_pot))
logpi_trace_HMC = vec(getPosition(MD_HMC, want_array=true, observable=interaction_pot))

plot(logpi_trace_UKLA, label="UKLA")
length(samples)
long_logpi_trace=Array{Float64}(undef, length(samples))
for iii=1:length(samples)
    long_logpi_trace[iii]=ℓπ(samples[iii])
end
plot!(long_logpi_trace, label="Advanced HMC")
display(plot!(logpi_trace_HMC, label="HMC"))

barycent(x) = mean(x)
mean_UKLA = vec(getPosition(MD_UKLA, want_array=true, observable=barycent))
mean_HMC = vec(getPosition(MD_UKLA, want_array=true, observable=barycent))
display(plot(mean_HMC, label="HMC barycenter"))
display(plot(mean_UKLA, label="UKLA barycenter"))
