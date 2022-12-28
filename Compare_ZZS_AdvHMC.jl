using Plots, BenchmarkTools, LinearAlgebra
using AdvancedHMC, Distributions, ForwardDiff
using JLD2  # For saving data
include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("functions_particles.jl")

N = 100 # number of particles
iter = 1 * 10^3 # number of iterations per thinned sample
thin_iter = 3 * 10^4 # number of thinned samples want to get. If ==1 then no thinning.
δ = 1e-2
# iter_UKLA = 6*10^5
# thin_iter_UKLA = 2 * 10^5
# δ_UKLA = 1e-5
n_samples, n_adapts = 1*10^2, 1*10^1
run_ZZS = false
run_advHMC = false
run_UKLA = false
plot_ZZS = false
plot_advHMC = false
plot_UKLA = false

want_variance = true
## Set up the target
a = 1.

V(r) = r^4
W(r) = - a * sqrt(1 + r^2)

Vprime(r) = 4 * (r^3)
Wprime(r) = -a * r / sqrt(1+r^2)

# V(r) = (1/r^12) - (1/r^6)
# W(r) = a * sqrt(1 + r^2)

# Vprime(r) = -12 * (1/r^13) + 6 * (1/r^7)
# Wprime(r) = a * r / sqrt(1+r^2)

# interaction_pot(x)=sum(V.(x[sortperm(x)[1:end-1]] - x[sortperm(x)[2:end]])) + 
#                     + sum(V.(x[sortperm(x)[1:end-2]] - x[sortperm(x)[3:end]])) +
#                     + sum(W.(x .- x')) / (2 * length(x))
interaction_pot(x)=sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end

Vrates(x,v,i)   = define_Vrates(Vprime, x, v, i, N)
Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j, N)

l = 1
x_init = randn(N)/l
# sort!(x_init)
v_init = rand((-1,1),N)

fun_var(x)=mean((x.-mean(x)).^2)
fun(x) = x.-mean(x)
# p_var = plot()

respath=string("particles/quartic")
if !isdir(respath)
    mkdir(respath)
end

## Run Zig-Zag 
if run_ZZS
    runtime_ZZS = @elapsed (chain_ZZS = splitting_zzs_particles(Vrates,Wrates,a,δ,N,iter,thin_iter,x_init,v_init));
    pos = getPosition(chain_ZZS)
    save_object(string(respath,"/zzs_a_",a,"_N_",N,"_delta_",δ,"_thiniter_",thin_iter,"_iter_",iter), pos)
    save_object(string(respath,"/zzs_runtime_a_",a,"_N_",N,"_delta_",δ,"_thiniter_",thin_iter,"_iter_",iter), runtime_ZZS)

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
    runtime_advHMC = @elapsed((samples, stats) = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; verbose=false, progress=true));
    pushfirst!(samples,x_init)

    save_object(string(respath,"/hmc_a_",a,"_N_",N,"_samples",n_samples), samples)
    save_object(string(respath,"/hmc_runtime_a_",a,"_N_",N,"_samples",n_samples), runtime_advHMC)
end

## Run UKLA
if run_UKLA
    fric = 1.0;
    η = exp(-δ * fric)
    K = 1
    v_init=randn(size(x_init))
    δ_UKLA = stats[end].step_size * 1e-9

    runtime_UKLA = @elapsed(MD_UKLA = UKLA(interaction_pot_grad, δ_UKLA, iter_UKLA, K, η, x_init, v_init,N_store=thin_iter_UKLA))
    pos_UKLA = getPosition(MD_UKLA)
    pl_UKLA = plot(reduce(hcat,fun.(pos_UKLA))',
        legend=:no, 
        title = "Trajectories UKLA",
    )
    display(pl_UKLA)
    emp_var_UKLA=Array{Float64}(undef, length(MD_UKLA))
    emp_var_UKLA[1]=fun_var(pos_UKLA[1])
    for iii=2:length(MD_UKLA)
        emp_var_UKLA[iii]=emp_var_UKLA[iii-1]+(fun_var(pos_UKLA[iii])-emp_var_UKLA[iii-1])/iii
    end

    p_var = plot!(p_var,emp_var_UKLA, label = "UKLA", 
            ylabel = "Empirical variance", xlabel = "Iterations",
            legend=:bottomright, linewidth=2)

    println("Run time for UKLA is $runtime_UKLA seconds")
    
end


## If want to load saved data
# pos = load_object(string(respath,"/zzs_a_0.01_N_25_delta_0.01_thiniter_100000_iter_1000"));
# samples = load_object(string(respath,"/hmc_a_0.01_N_25_samples80000"));
runtime_ZZS = load_object(string(respath,"/zzs_runtime_a_1.0_N_100_delta_0.01_thiniter_30000_iter_1000"));
runtime_advHMC = load_object(string(respath,"/hmc_runtime_a_1.0_N_100_samples100"));
# a = 0.01
# N = 25

## Plots variance

p_var = plot();
x_max = 850;
if plot_advHMC
    var_advHMC=Array{Float64}(undef, length(samples))
    var_advHMC[1]=fun_var(samples[1])
    for iii=2:length(samples)
        var_advHMC[iii]=var_advHMC[iii-1]+(fun_var(samples[iii])-var_advHMC[iii-1])/iii
    end
    runtimes_hmc = [runtime_advHMC * i/length(samples) for i=0:length(samples)-1]
    p_var = plot!(p_var,runtimes_hmc,var_advHMC, 
                # label="HMC", 
                label = "",
                ylabel = "", xlabel = "", 
                # ylabel = "Empirical variance", xlabel = "Runtime (seconds)", 
                guidefontsize=15,
                tickfontsize = 13,
                xlims = [-0.10*x_max,x_max],
                # xticks = [0,50,100,150,200],
                # xticks = [0,25,50,75],
                ylims = [100,2400],
                # yaxis=:log,
                linecolor =:red,
                legend=:bottomright, linewidth=4);
end

# hline!([var_advHMC[end]],line=:dash,color=:black,label="", linewidth=1.5)

if plot_ZZS
        emp_var=Array{Float64}(undef, length(pos))
        emp_var[1]=fun_var(pos[1])
        # for iii=2:length(pos)
        #     emp_var[iii]=emp_var[iii-1]+(fun_var(pos[iii])-emp_var[iii-1])/iii
        # end
        for iii in eachindex(pos[2:end])
            emp_var[iii+1]=emp_var[iii]+(fun_var(pos[iii+1])-emp_var[iii])/(iii+1)
        end
        runtimes_zzs = [runtime_ZZS * i/(size(pos,1)-1) for i=0:(size(pos,1)-1)]
        p_var = plot!(p_var,runtimes_zzs,emp_var, 
                # label = "Unadjusted ZZS", 
                label = "",
                # ylabel = "Empirical variance", xlabel = "Runtime (seconds)", 
                linecolor =:green,
                # line=:dot,
                legend=:bottomright, legendfontsize=10, linewidth=4);
end
# if plot_UKLA
#     p_var = plot!(p_var,emp_var_UKLA, label = "UKLA", 
#                 ylabel = "Empirical variance", xlabel = "Iterations",
#                 legend=:bottomright, linewidth=2)
# end
display(p_var)
# savefig(p_var, string(respath,"/empvar_a",a,"_itersperthinzzs_",iter,".pdf"))
# savefig(p_var, string(respath,"/empvar_a",a,"_N_",N,"_itersperthinzzs_",iter,"_nolegend.pdf"))


println("")
println("Run time for AdvancedHMC is $runtime_advHMC seconds")
println("Run time for ZZS is $runtime_ZZS seconds")
if want_variance
println("Estimated variance around barycentre ZZS:", emp_var[end])
println("Estimated variance around barycentre HMC:", var_advHMC[end])
end
# println("Estimated diameter ZZS:", diam_zzs[end])
# println("Estimated diameter HMC:", diam_hmc[end])
println("Step size ZZS: $δ")
println("Step size HMC: ", stats[end].step_size)


# Trajectories 

a=0.01
pos = load_object(string(respath,"/zzs_a_0.01_N_25_delta_0.01_thiniter_100000_iter_1000"));
samples = load_object(string(respath,"/hmc_a_0.01_N_25_samples80000"));

a = 10
pos = load_object(string(respath,"/zzs_a_10.0_N_25_delta_0.01_thiniter_100000_iter_1000"));
samples = load_object(string(respath,"/hmc_a_10.0_N_25_samples100000"));
iters_plot = 10^4;

a = 1
pos = load_object(string(respath,"/zzs_a_1.0_N_100_delta_0.01_thiniter_30000_iter_1000"));
samples = load_object(string(respath,"/hmc_a_1.0_N_100_samples100"));
iters_plot = 10^3;
# pl = plot(reduce(hcat,fun.(pos[1:iters_plot]))',
#         legend=:no, 
#         # title = "Centred trajectories Unadjusted ZZS",
# #  xlims = [0,100],
#     )
# pl = plot(reduce(hcat,fun.(pos[1:10*iters_plot]))',
#         legend=:no, 
#         # title = "Centred trajectories Unadjusted ZZS",
# #  xlims = [0,100],
#     )
# display(pl)
# savefig(pl, string("traj_centred_zzs_a",a,"_N_",N,"_delta_",δ,"_itersperthinzzs_",iter,".pdf"))

plot(reduce(hcat,fun.(pos[1:10^2]))',
    legend=:no, 
    title = "Trajectories Unadjusted ZZS a=$a",
#  xlims = [0,100],
)

pl_HMC = plot(reduce(hcat,samples[9*iters_plot:10*iters_plot])',
    legend=:no, 
    title = "Trajectories HMC",
     #  xlims = [0,100],
    )
pl_HMC = plot(reduce(hcat,fun.(samples[(1+0*iters_plot):100]))[1:5,:]',
    legend=:no, 
    title = "Trajectories HMC a=$a",
     #  xlims = [0,100],
    )
pl_HMC = plot!(reduce(hcat,fun.(samples[(1+0*iters_plot):100]))[25:30,:]',
    legend=:no, 
    title = "Trajectories HMC a=$a",
     #  xlims = [0,100],
    )
    pl_HMC = plot!(reduce(hcat,fun.(samples[(1+0*iters_plot):100]))[50:55,:]',
    legend=:no, 
    title = "Trajectories HMC a=$a",
     #  xlims = [0,100],
    )
N=100
scatter(reduce(hcat,1:N), pos[1]',
        legend=:no,     )
scatter(reduce(hcat,1:N), pos[5]',
        legend=:no,     )
scatter(reduce(hcat,1:N), pos[10]',
        legend=:no,     )
scatter(reduce(hcat,1:N), pos[100]',
        legend=:no,     )
scatter(reduce(hcat,1:N), pos[end]',
        legend=:no,     )

        anim = @animate for i ∈ 1:100
            scatter(reduce(hcat,1:N), fun(pos[i])',
        legend=:no,  
        title = "UZZS: iteration $i",
        xlabel = "particles",
        ylabel = "height",
        ylims =  [-85,85]  )
        end
        gif(anim, "zzs_particles_100.gif", fps = 2)

        anim = @animate for i ∈ 1:20
            scatter(reduce(hcat,1:N), fun(samples[i])',
        legend=:no,  
        title = "HMC: iteration $i",
        xlabel = "particles",
        ylabel = "height",
        ylims =  [-100,100]  )
        end
        gif(anim, "hmc_particles.gif", fps = 1)


iters_plot


# pl_HMC = plot(reduce(hcat,fun.(samples[1:200]))[2:3,:]',
#     legend=:no, 
#     # title = "Trajectories HMC",
#      #  xlims = [0,100],
#     )
# display(pl_HMC)
# savefig(pl_HMC, string("traj_centred_hmc_a",a,"_N_",N,".pdf"))

# p_bary = plot(mean.(pos), label = "Unadjusted ZZS", 
#             # xlims = [-400,thin_iter+700],
#             ylabel = "Barycentre", xlabel = "Iterations", 
#             linecolor =:green,
#             legend=:bottomright, legendfontsize=10, linewidth=1)
# p_bary = plot!(p_bary, mean.(samples),label="HMC", 
#             linecolor =:red,
#             legend=:bottomright, linewidth=1)
# savefig(p_bary, string("barycentre_a",a,"_N_",N,"_delta_",δ,"_itersperthinzzs_",iter,".pdf"))

        
# p_bar = plot(mu', label="baricentre")
# savefig(p_bar, string("baricentre_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))
# trace_potential = [interaction_pot(pos[:,i]) for i=1:length(chain_ZZS)]
# p_tracepot = plot(trace_potential, label="Trace potential", yaxis=:log)
# savefig(p_tracepot, string("tracepot_a",a,"_thin_",thin_iter,"_itersperthin_",iter,".pdf"))

## Plots diameter = maximum - minimum 
# pos = load_object(string(respath,"/zzs_a_1.0_N_10_delta_0.01"))
# samples = load_object(string(respath,"/hmc_a_1.0_N_10"))
# runtime_ZZS = load_object(string(respath,"/zzs_runtime_a_1.0_N_10_delta_0.01"))
# runtime_advHMC = load_object(string(respath,"/hmc_runtime_a_1.0_N_10"))

# fun_diam(x) = maximum(x)-minimum(x);

# p_diam = plot();
# if plot_ZZS
#     diam_zzs=Array{Float64}(undef, size(pos,1))
#     diam_zzs[1]=fun_diam(pos[1])
#     for iii=2:size(pos,1)
#         diam_zzs[iii]=diam_zzs[iii-1]+(fun_diam(pos[iii])-diam_zzs[iii-1])/iii
#     end
#     runtimes_zzs = [runtime_ZZS * i/(size(pos,1)-1) for i=0:(size(pos,1)-1)]
#     p_diam = plot!(p_diam,runtimes_zzs,diam_zzs, label = "Unadjusted ZZS", 
#             # xlims = [-400,thin_iter+700],
#             ylabel = "Diameter", xlabel = "Run times (seconds)", 
#             linecolor =:green,
#             legend=:bottomright, legendfontsize=10, linewidth=2);
#     p_diam = scatter!(p_diam,[runtimes_zzs[end]],[diam_zzs[end]], label="", 
#             # seriestype=:scatter,
#             markershape=:xcross,
#             markersize=:5,
#             markerstrokewidth=3,
#             color =:green,
#             legend=:bottomright, 
#             );
# end
# if plot_advHMC
#     diam_hmc=Array{Float64}(undef, length(samples))
#     diam_hmc[1]=fun_diam(samples[1])
#     for iii=2:length(samples)
#         diam_hmc[iii]=diam_hmc[iii-1]+(fun_diam(samples[iii])-diam_hmc[iii-1])/iii
#     end
#     runtimes_hmc = [runtime_advHMC * i/length(samples) for i=0:length(samples)-1]
#     p_diam = plot!(p_diam,runtimes_hmc,diam_hmc, label="HMC", 
#             ylabel = "Diameter", xlabel = "Runtime (seconds)", 
#             linecolor =:red,
#             legend=:bottomright, linewidth=2);
#     p_diam = scatter!(p_diam,[runtimes_hmc[end]],[diam_hmc[end]], label="", 
#             # seriestype=:scatter,
#             markershape=:xcross,
#             markersize=:5,
#             markerstrokewidth=3,
#             color =:red,
#             legend=:bottomright, 
#             # yaxis=:log,
#             );
# end
# display(p_diam)
# # savefig(p_diam, string("diam_a",a,"_itersperthinzzs_",iter,".pdf"))
