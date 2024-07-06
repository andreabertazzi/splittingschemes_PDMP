## Compute the error of splitting schemes for some test functions for different refreshment rates.
# The goal is to check the behaviour as a function of the refreshment rates and see
# if it is as predicted by the theory.

include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")
include("run_experiments.jl")


## Chose the relevant parameters
dim = 1
δ = 1 * 10^(-1)
T = 1 * 10^5
N = Int64(round(T / δ))

# first_part = LinRange(0.0, 0.2, 2)
# # first_part = LinRange(0.05, 0.2, 4)
# second_part = LinRange(0.4, 3.0, 10)
# refresh_rates = [first_part; second_part] # this makes it a vector containing the right values
# refresh_rates = LinRange(0.05, 3.0, 12)
refresh_rates = LinRange(0.05, 3.0, 10)


n_exp = 50

samplers = [
    splitting_bps_DBRBD_fun,
    splitting_bps_RDBDR_fun,
    splitting_bps_DRBRD_fun,
    splitting_bps_BDRDB_fun,
]
## For Gaussian target
# ρ = 0.0
# Σ = ρ * ones(dim, dim) + (1 - ρ)I
# # Σ[1,1] = 0.05
# Σ_inv = inv(Σ)
# Σ_sqrt = sqrt(Σ)
# μ = zeros(dim)
# ∇U(x) = Σ_inv * (x - μ)
# true_rad = Float64(sum(diag(Σ)))
# truth = true_rad
# radius(x) = sum(x .^ 2)
# test_func = radius

# function initial_state_gauss(dim::Int64)
#     x = Σ_sqrt * randn(dim)
#     v = randn(dim)
#     v = v/norm(v)
#     (x,v)
# end

# errors = run_by_refreshmentrates(
#     ∇U,
#     samplers,
#     initial_state_gauss,
#     # initial_state_bps_gaussianpos,
#     dim,
#     δ,
#     N,
#     refresh_rates,
#     n_exp,
#     test_func,
#     truth,
# )

## For Cauchy distribution in 1D
# ∇U(x) = 2*x./(1 + norm(x)^2)
# c = 2
# truncatedradius = (pi * c + 2 * sqrt(c) - 2 * (c+1) * atan(sqrt(c))) / pi
# truth = truncatedradius
# truncated_radius(x) = min(sum(x.^2),c)
# # test_func = Vector{Function}(undef,1)
# test_func = truncated_radius
# using Distributions

# function initial_state_cauchy(dim::Int64)
#     distrib = Cauchy()
#     draw_x = rand(distrib)
#     x = [draw_x]
#     v = randn(dim)
#     v = v/norm(v)
#     (x,v)
# end
# errors = run_by_refreshmentrates(
#     ∇U,
#     samplers,
#     initial_state_cauchy,
#     dim,
#     δ,
#     N,
#     refresh_rates,
#     n_exp,
#     test_func,
#     truth,
# )


## For the case U(x)=x^4 ##

∇U(x) = 4 * x.^3
radius(x) = sum(x .^ 2)
test_func = radius
using QuadGK
fn(x) = exp(-x^4)
partition_fn, err = quadgk(x -> fn(x), -Inf, Inf, rtol=1e-8)
prob_dens(x) = fn(x)/partition_fn
tf(x) = test_func(x) * prob_dens(x)
expest,err = quadgk(x -> tf(x), -Inf, Inf, rtol=1e-8)
truth = expest
errors = run_by_refreshmentrates(
    ∇U,
    samplers,
    initial_state_bps_gaussianpos,
    dim,
    δ,
    N,
    refresh_rates,
    n_exp,
    test_func,
    truth,
)


## Compute errors and do plots

# create_matrix_errors!(avg_err, errors, length(samplers))  # take averages

avg_err = mean(errors; dims = 2)
avg_err = dropdims(avg_err, dims = tuple(findall(size(avg_err) .== 1)...))
avg_err = transpose(avg_err)

st_devs_err = sqrt.(var(errors, dims = 2))
st_devs_err = dropdims(st_devs_err, dims =tuple(findall(size(st_devs_err).==1)...))
st_devs_err = transpose(st_devs_err)

colours = [:red, :green, :orange, :blue]

q = plot(
    refresh_rates,
    # avg_err[:, 1],
    sqrt.(avg_err[:, 1]),
    ribbon=st_devs_err[:,1],
    fillalpha=.3,
    fillcolor = colours[1],
    label = "Splitting DBRBD",
    # q = plot(refresh_rates,avg_1, #yaxis = :log, xaxis=:log,
    linecolor = colours[1],
#     ylabel = "Root MSE for radius statistic",
    ylabel = "Root MSE for truncated radius",
    xlabel = "Refreshment rate",
    # line = :solid,
    # linestyle=:dash,
    linewidth = 2,
    # title = title_plot,
    legend = :topleft,
    legendfontsize = 10,
    xticks = [0, 0.5, 1, 1.5, 2, 2.5, 3],
    # xticks = [0, 0.25, 0.5, 0.75, 1],
    # xticks = (δ,[0.975,0.95,0.925,0.9]),
    # ylims = [0, 1],
    # yticks=[10^0.5,10^0,10^(-0.5),10^(-1),10^(-1.5),10^(-2)],
)
q = plot!(
    refresh_rates,
    # avg_err[:, 2],
    sqrt.(avg_err[:, 2]),
    ribbon=st_devs_err[:,2],
    fillalpha=.3,
    fillcolor = colours[2],
    label = "Splitting RDBDR",
    linewidth = 2,
    linecolor = colours[3],
)
q = plot!(
    refresh_rates,
    # avg_err[:, 3],
    sqrt.(avg_err[:, 3]),
    ribbon=st_devs_err[:,3],
    fillalpha=.3,
    fillcolor = colours[3],
    label = "Splitting DRBRD",
    linewidth = 2,
    linecolor = colours[2],
)
q = plot!(
    refresh_rates,
    # avg_err[:, 4],
    sqrt.(avg_err[:, 4]),
    ribbon=st_devs_err[:,4],
    fillalpha=.3,
    fillcolor = colours[4],
    label = "Splitting BDRDB",
#     linestyle=:dash,
    linewidth = 2,
    linecolor = colours[4],
)

display(q)


# # Save figure in Gaussian case
# # savefig(string("bps_empi_err_gauss_1d_delta_",δ,"_nexp_",n_exp,"_w_o_Euler.pdf"))
# # savefig(string("bps_empi_err_gauss_dim_",dim,"_delta_",δ,"_nexp_",n_exp,".pdf"))
# savefig(string("bps_empi_err_gauss_dim_",dim,"_rho_",ρ,"_delta_",δ,"_nexp_",n_exp,".pdf"))

# # Save non Lipschitz
# savefig(string("bps_empi_err_nonlipsch_1d_delta_",δ,"_nexp_",n_exp,"horizon",T,".pdf"))

# # Save figure in Cauchy case
# savefig(string("bps_empi_err_cauchy_1d_delta_",δ,"_nexp_",n_exp,"horizon_",T,"_c=",c,".pdf"))
# q

## To save the data
# using DataFrames, StatsPlots, CSV
# function add_entries_errors!(df::DataFrame, errs::Array{Float64},
#                     name_sampler::Vector{String}, rr)
#           for j = 1 : length(name_sampler)
#             for (i,rates) in enumerate(rr)
#             push!(
#                   df,
#                   Dict(
#                       :sampler => name_sampler[j],
#                       :refresh_rate => rates,
#                       :error_radius => errs[i,j]
#                       ),
#                   )
#             end
#           end
# end
# df = DataFrame(
#     sampler = String[],
#     refresh_rate = Float64[],
#     error_radius = Float64[],
# );
# names_samplers = Vector{String}()
# push!(names_samplers,"Splitting DBRBD")
# push!(names_samplers,"Splitting RDBDR")
# push!(names_samplers,"Splitting DRBRD")
# push!(names_samplers,"Splitting BDRDB")
# # string_names_samplers!(names_samplers)
# add_entries_errors!(df,avg_err,names_samplers,refresh_rates)
# CSV.write(string("bps_empi_MSE_cauchy_d_", dim,"_c_", c,"_delta_",δ,"_nexp_",n_exp,"_dimension", dim,"-horizon-",T,".csv"), df)
# CSV.write(string("bps_empi_MSE_gauss_d_", dim,"_rho_", ρ,"_delta_",δ,"_nexp_",n_exp,"_dimension", dim,"-horizon-",T,".csv"), df)

# # To read the data
# df1 = CSV.read("Plots_new_expansion_invmeas/Compare all schemes/bps_empi_err_gauss_1d_delta_1.0_nexp_250_dimension1-horizon-100000.csv", DataFrame)

# # avg_err_DBRBD = df1[df1.sampler .== "Splitting BRDRB", 3]
# names_samplers = Vector{String}(undef,10)
# string_names_samplers!(names_samplers)

# function read_data_give_errors(df::DataFrame, names_samplers::Vector{String})
#     n_rr = sum(df1.sampler .== names_samplers[1])
#     n_samplers = length(names_samplers)
#     avg_err = Array{Float64,2}(undef, n_rr, n_samplers)
#     for i = 1:n_samplers
#         avg_err[:,i] = df1[df1.sampler .== names_samplers[i], 3]
#     end
#     avg_err
# end

# avg_err = read_data_give_errors(df1,names_samplers)
# refresh_rates = df1[df1.sampler .== names_samplers[1], 2]





## Plots for all splittings
# samplers = [
#             splitting_bps_BRDRB_fun,
#             splitting_bps_DRBRD_fun,
#             splitting_bps_BDRDB_fun,
#             splitting_bps_RDBDR_fun,
#             splitting_bps_DBRBD_fun,
#             splitting_bps_RBDBR_fun,
#             splitting_bps_DBD_fun,
#             splitting_bps_BDB_fun,
#             splitting_bps_DR_B_DR_fun,
#             splitting_bps_B_DR_B_fun
#             ]

# errors = run_by_refreshmentrates(
#             ∇U,
#             samplers,
#             initial_state_bps,
#             dim,
#             δ,
#             N,
#             refresh_rates,
#             n_exp,
#             test_func,
#             truth
#             )
# # #
# avg_err = Array{Float64,2}(undef, length(refresh_rates), length(samplers))
# create_matrix_errors!(avg_err, errors, length(samplers))
# colours = [:orchid1,:red,:green,:orange]


# q = plot(refresh_rates,
#         sqrt.(avg_err[:,6]),
#         # avg_err[:,6],
#           label = "Splitting RBDBR",
# # q = plot(refresh_rates,avg_1, #yaxis = :log, xaxis=:log,
#           linecolor=colours[2],
#           ylabel = "Root MSE for radius statistic",
#           xlabel = "Refreshment rate",
#           line=:solid,
#           linewidth=2,
#           # title = title_plot,
#           legend=:topleft,
#           legendfontsize=10,
#           xticks=[0,0.5,1,1.5,2,2.5,3],
#           # xticks = (δ,[0.975,0.95,0.925,0.9]),
#           ylims= [-.05,1],
#           # yticks=[10^0.5,10^0,10^(-0.5),10^(-1),10^(-1.5),10^(-2)],
#           )
# q = plot!(refresh_rates,
#             # avg_err[:,3],
#             sqrt.(avg_err[:,3]),
#             label = "Splitting BDRDB",linewidth=2,linecolor=colours[4])
# q = plot!(refresh_rates,
#             # avg_err[:,1],
#             sqrt.(avg_err[:,1]),
#             label = "Splitting BRDRB",linewidth=2,linecolor=colours[3])
# q = plot!(refresh_rates,
#             # avg_err[:,8],
#             sqrt.(avg_err[:,8]),
#             label = "Splitting BDB",linewidth=2,linecolor=colours[1])
# q = plot!(refresh_rates,
#             # avg_err[:,10],
#             sqrt.(avg_err[:,10]),
#             label = "Splitting B_DR_B",linewidth=2,linecolor="black")
# # savefig(string("bps_empi_err_gauss_1d_delta_",δ,"_nexp_",n_exp,"_all_scarsi.pdf"))
# # savefig(string("bps_empi_err_gauss_1d_delta_",δ,"_nexp_",n_exp,"_scarsi.pdf"))


# r = plot(refresh_rates,
#         # avg_err[:,5],
#         sqrt.(avg_err[:,5]),
#         label = "Splitting DBRBD",
#           #yaxis = :log, xaxis=:log,
#           # linecolor=colours[2],
#           ylabel = "Root MSE for radius statistic",
#           xlabel = "Refreshment rate",
#           line=:solid,
#           linewidth=2,
#           linecolor=colours[2],
#           # title = title_plot,
#           legend=:topleft,
#           legendfontsize=10,
#           xticks=[0,0.5,1,1.5,2,2.5,3],
#           # xticks = (δ,[0.975,0.95,0.925,0.9]),
#           ylims= [-0.05,1],
#           # yticks=[10^0.5,10^0,10^(-0.5),10^(-1),10^(-1.5),10^(-2)],
#           )
# # r = plot(refresh_rates,avg_2,label = "Splitting DRBRD",linewidth=2)
# r = plot!(refresh_rates,
#             # avg_err[:,4],
#             sqrt.(avg_err[:,4]),
#             label = "Splitting RDBDR",linewidth=2,linecolor=colours[4])
# r = plot!(refresh_rates,
#         # avg_err[:,2],
#         sqrt.(avg_err[:,2]),
#         label = "Splitting DRBRD",linewidth=2,linecolor=colours[3])
# r = plot!(refresh_rates,    
#             # avg_err[:,7],
#             sqrt.(avg_err[:,7]),
#             label = "Splitting DBD",linewidth=2,linecolor=colours[1])
# r = plot!(refresh_rates,
#             # avg_err[:,9],
#             sqrt.(avg_err[:,9]),
#             label = "Splitting DR_B_DR",linewidth=2, linecolor="black")
# savefig(string("bps_empi_err_gauss_1d_delta_",δ,"_nexp_",n_exp,"_all_forti.pdf"))
# savefig(string("bps_empi_err_gauss_1d_delta_",δ,"_nexp_",n_exp,"_forti.pdf"))


# avg_err_euler = transpose(mean(errors[1,:,:,1]; dims=1))
# avg_err_split_ABCBA = transpose(mean(errors[2,:,:,1]; dims=1))
# avg_err_split_ACBCA = transpose(mean(errors[3,:,:,1]; dims=1))
# avg_err_split_BACAB = transpose(mean(errors[4,:,:,1]; dims=1))
#
# colours = [:blue,:red,:green,:orange]
# q = plot(refresh_rates,avg_err_split_ABCBA, # yaxis = :log, xaxis=:log,
#           linecolor=colours[2],
#           label = "Splitting DBRBD",
#           ylabel = "Error radius",
#           xlabel = "Refreshment rate",
#           line=:solid,
#           linewidth=2,
#           # title = title_plot,
#           legend=:topleft,
#           legendfontsize=10,
#           xticks=[0,0.5,1,1.5,2,2.5,3],
#           # xticks = (δ,[0.975,0.95,0.925,0.9]),
#           #ylims=(0,avg_err_rad_split_ACBCA[end]),
#           # yticks=[10^0.5,10^0,10^(-0.5),10^(-1),10^(-1.5),10^(-2)],
#           )
# q = plot!(refresh_rates,avg_err_split_BACAB,label = "Splitting BDRDB",linecolor=colours[4],linewidth=2)
# q = plot!(refresh_rates,avg_err_split_ACBCA,label = "Splitting DRBRD",linecolor=colours[3],linewidth=2)
# q = plot!(refresh_rates,avg_err_euler,label = "Euler",linecolor=colours[1],linewidth=2)

# Save figure in Gaussian case
# savefig(string("bps_empi_err_gauss_1d_delta_",δ,"_nexp_",n_exp,"_w_o_Euler.pdf"))
# savefig(string("bps_empi_err_gauss_1d_delta_",δ,"_nexp_",n_exp,"_wEuler.pdf"))

# savefig(string("bps_empi_err_nonlipsch_1d_delta_",δ,"_nexp_",n_exp,"w_o_Euler.pdf"))
# savefig(string("bps_empi_err_nonlipsch_1d_delta_",δ,"_nexp_",n_exp,"wEuler.pdf"))

# Save figure in Cauchy case
# savefig(string("bps_empi_err_cauchy_1d_delta_",δ,"_nexp_",n_exp,"_w_o_Euler.pdf"))
# savefig(string("bps_empi_err_cauchy_1d_delta_",δ,"_nexp_",n_exp,"_wEuler.pdf"))

## If in addition want to check error dependence for continuous time process
# samplers_cts = Vector{Function}(undef,1)
# samplers_cts[1] = BPS
# discr_step = 0.05
# err_bps = estimate_error_byrefresh_cts(
#                 ∇U,
#                 Σ_inv,
#                 samplers_cts,
#                 initial_state_bps,
#                 T,
#                 discr_step,
#                 refresh_rates,
#                 n_exp,
#                 test_func,
#                 truth
#                 )
#
# avg_err_bps = transpose(mean(err_bps[1,:,:,1]; dims=1))
# q = plot(refresh_rates,avg_err_bps, # yaxis = :log, xaxis=:log,
#           linecolor=colours[2],
#           label = "BPS",
#           ylabel = "Error radius",
#           xlabel = "Refreshment rate",
#           line=:solid,
#           linewidth=2,
#           # title = title_plot,
#           legend=:topleft,
#           legendfontsize=10,
#           xticks=[0,0.5,1,1.5,2,2.5,3],
#           # xticks = (δ,[0.975,0.95,0.925,0.9]),
#           #ylims=(0,avg_err_rad_split_ACBCA[end]),
#           # yticks=[10^0.5,10^0,10^(-0.5),10^(-1),10^(-1.5),10^(-2)],
#           )
