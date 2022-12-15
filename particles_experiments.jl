using Plots
# using JLD2  # For saving data
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("imaging_utils.jl")


N = 5 # number of particles
iter = 10^7 # number of iterations
δ = 1e-3

a = 10.0
V(r) = (1/r^12) - (1/r^6)
W(r) = a * sqrt(1 + r^2)

Vprime(r) = -12 * (1/r^13) + 6 * (1/r^7)
Wprime(r) = a * r / sqrt(1+r^2)

function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
end

function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end

Vrates(x,v,i)   = define_Vrates(Vprime, x, v, i, N)
Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j, N)

l = 1
x_init = (collect(1:N).-N/2)./l
v_init = rand((-1,1),N)
chain_ZZS = splitting_zzs_particles(Vrates,Wrates,a,δ,N,iter,x_init,v_init);
p = plot()
# mu = zeros(iter+1)
pos = getPosition(chain_ZZS; want_array=true)
mu = mean(pos; dims = 1)
trace_potential = [interaction_pot(pos[:,i]) for i=1:iter+1]
plot(trace_potential, label="Trace potential")
plot(mu, label="baricentre")

p = plot()
for j = 1:N
    positions = [chain_ZZS[i].position[j] for i = 1:(iter+1)];
    # p = plot!(positions)
    c = positions - mu'
    p = plot!(c)
    # pos[j,:] -= mu'
end
# mu = mu./N
display(p)


# positions1 = [chain_ZZS[i].position[1] for i = 1:iter];
# positions2 = [chain_ZZS[i].position[2] for i = 1:iter];
# positions3 = [chain_ZZS[i].position[3] for i = 1:iter];
# positions4 = [chain_ZZS[i].position[4] for i = 1:iter];
# positions5 = [chain_ZZS[i].position[5] for i = 1:iter];

# plot(positions1)
# plot!(positions2)
# plot!(positions3)
# plot!(positions4)
# plot!(positions5)

# positionsl = [chain_ZZS[i].position[50] for i = 1:iter];
# plot(positionsl)
# plot(positions2)
# plot(positions1.-positions2)