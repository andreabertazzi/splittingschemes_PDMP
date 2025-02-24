using Plots
# using JLD2  # For saving data
include(joinpath(@__DIR__, "..", "helper_split.jl"))
include(joinpath(@__DIR__, "..","moves_samplers.jl"))
include(joinpath(@__DIR__, "..","algorithms.jl"))
include(joinpath(@__DIR__, "..","functions_particles.jl"))

N = 25 # number of particles
iter = 1 * 10^3 # number of iterations per thinned sample
thin_iter = 10^4 # number of thinned samples want to get. If ==1 then no thinning.
δ = 1e-2

a = 1.

V(r) = r^4
W(r) = - a * sqrt(1 + r^2)

Vprime(r) = 4 * (r^3)
Wprime(r) = -a * r / sqrt(1+r^2)

# Define rates for ZZP
# Vrates(x,v,i)   = define_Vrates(Vprime, x, v, i, N)
# Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j, N)
# Define gradients for BPS
# Vgrad(x)        = define_Vgrad(Vprime, x, N)
# Wgrad(x,j)      = define_Wgrad(Wprime, x, j, N)
function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end
function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
end
# Vgrad(x) = vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])))
# Wgrad(x,j) = vec(Wprime.(x .- x[j]) / (length(x)))

l = 1
b = 1
# nr_top = convert(Int,ceil(N/2))
# x_init = zeros(N)
# x_init[1:nr_top] = b .+ collect(1:nr_top)/l
# x_init[nr_top+1:N] = -b .- collect((nr_top+1):N)/l
# sort!(x_init)  # important step as else the LJ potential does not work well
# x_init = (collect(1:N).-N/2)./l
x_init = randn(N)/l
sort!(x_init)
# v_init = rand((-1,1),N)
# runtime_ZZS = @elapsed ( 
#     chain_ZZS = splitting_zzs_particles(Vrates,Wrates,a,δ,N,iter,thin_iter,x_init,v_init)
#     )
v_init = randn(N)
ub = a
refresh_rate = 10.
chain_ZZS = splitting_bps_particles(ub,δ,N,iter,thin_iter,x_init,v_init)
# chain_ZZS = splitting_bps_particles(Vprime,Wprime,ub,δ,N,iter,thin_iter,x_init,v_init)

# chain_ZZS = splitting_bps_particles_full(interaction_pot_grad,δ,iter,thin_iter,x_init,v_init)

# pos = getPosition(chain_ZZS; want_array=true)
# mu = mean(pos; dims = 1)
# plot(mu', label="baricentre")

pos = [chain_ZZS[i].position for i in eachindex(chain_ZZS)]
# fun(x) = x.-mean(x)
# plot(reduce(hcat,fun.(pos))',
#     legend=:no, 
#     title = "Trajectories",
# #  xlims = [0,100],
#     )

# pl = plot()
# for j = 1:N
#     positions = [chain_ZZS[i].position[j] for i = 1:length(chain_ZZS)];
#     global pl = plot!(positions, 
#      legend=:no, 
#      title = "Trajectories",
#     #  ylims = [-3,3],
#      )
#     # c = positions - mu'
#     # global pl = plot!(c, legend=:no, title="Trajectories subtracting the baricentre", 
#             # ylims=[-4,4],
#             # )
# end

# display(pl)
# savefig(pl, string("trajectories_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))


# p_bar = plot(mu', label="baricentre")
# savefig(p_bar, string("baricentre_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))
# trace_potential = [interaction_pot(pos[:,i]) for i=1:length(chain_ZZS)]
# p_tracepot = plot(trace_potential, label="Trace potential", yaxis=:log)
# savefig(p_tracepot, string("tracepot_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))

fun_var(x)=mean((x.-mean(x)).^2)
emp_var=Array{Float64}(undef, length(chain_ZZS))
emp_var[1] = fun_var(pos[1])
# emp_var[2:end] = [emp_var[iii+1]+(fun_var(pos[iii+1])-emp_var[iii+1])/(iii+1) for iii in eachindex(chain_ZZS[2:end])]
# for iii=2:length(chain_ZZS)
#     emp_var[iii]=emp_var[iii-1]+(fun_var(pos[iii])-emp_var[iii-1])/iii
# end
for iii in eachindex(pos[2:end])
    emp_var[iii+1]=emp_var[iii]+(fun_var(pos[iii+1])-emp_var[iii])/(iii+1)
end
# emp_var = compute_variance_particles(chain_ZZS, mean.(pos))
plot(emp_var, label = "ZZS", ylabel = "Empirical variance", xlabel = "Thinned iterations")

# v(x) = (1/N^2) * sum((x.-transpose(x)).^2)
# emp_var=Array{Float64}(undef, length(chain_ZZS))
# emp_var[1]=v(pos[1])
# for iii=2:length(chain_ZZS)
#     emp_var[iii]=emp_var[iii-1]+(v(pos[iii])-emp_var[iii-1])/iii
# end
# # emp_var = compute_variance_particles(chain_ZZS, mean.(pos))
# plot(emp_var, label = "ZZS", ylabel = "Empirical variance", xlabel = "Thinned iterations")


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
