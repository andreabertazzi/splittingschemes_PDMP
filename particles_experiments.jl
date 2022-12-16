using Plots
# using JLD2  # For saving data
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("functions_particles.jl")

N = 10 # number of particles
iter = 1 * 10^3 # number of iterations per thinned sample
thin_iter = 10^4 # number of thinned samples want to get. If ==1 then no thinning.
δ = 1e-4

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
b = 1
# nr_top = convert(Int,ceil(N/2))
# x_init = zeros(N)
# x_init[1:nr_top] = b .+ collect(1:nr_top)/l
# x_init[nr_top+1:N] = -b .- collect((nr_top+1):N)/l
# sort!(x_init)  # important step as else the LJ potential does not work well
# x_init = (collect(1:N).-N/2)./l
x_init = randn(N)/l
sort!(x_init)
v_init = rand((-1,1),N)
chain_ZZS = splitting_zzs_particles(Vrates,Wrates,a,δ,N,iter,thin_iter,x_init,v_init);
# chain_ZZS = thinned_splitting_zzs_particles(Vrates,Wrates,a,δ,N,thin_iter,iter,x_init,v_init);


# pos = getPosition(chain_ZZS; want_array=true)
# mu = mean(pos; dims = 1)
# plot(mu', label="baricentre")

pos = getPosition(chain_ZZS)
fun(x) = x.-mean(x)
plot(reduce(hcat,fun.(pos))',
    legend=:no, 
    title = "Trajectories",
#  xlims = [0,100],
    )

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
emp_var[1]=fun_var(pos[1])
for iii=2:length(chain_ZZS)
    emp_var[iii]=emp_var[iii-1]+(fun_var(pos[iii])-emp_var[iii-1])/iii
end
# emp_var = compute_variance_particles(chain_ZZS, mean.(pos))
plot(emp_var, label = "ZZS", ylabel = "Empirical variance", xlabel = "Thinned iterations")


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
