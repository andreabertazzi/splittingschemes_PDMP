df1 = CSV.read("bps_empi_MSE_gauss_d_10_rho_0.7_delta_0.5_nexp_200_dimension10-horizon-100000.csv", DataFrame)

dim = 10
ρ = 0.7
δ = 0.5 
n_exp = 200


function read_data_give_errors(df::DataFrame)
    n_rr = sum(df.sampler .== df[1,:].sampler)
    # n_rr = sum(df.sampler .== df[1,1])
    refresh_rates = unique(df.refresh_rate)
    names_samplers = unique(df.sampler)
    n_samplers = length(names_samplers)
    avg_err = Array{Float64,2}(undef, n_rr, n_samplers)
    for i = 1:n_samplers
        avg_err[:,i] = df[df.sampler .== names_samplers[i], 3]
    end
    (refresh_rates,avg_err)
end

(refresh_rates,avg_errors) = read_data_give_errors(df1)

colours = [:red, :green, :orange, :blue]

q = plot(
    refresh_rates,
    # avg_err[:, 1],
    sqrt.(avg_errors[:, 1]),
    label = "Splitting DBRBD",
    # q = plot(refresh_rates,avg_1, #yaxis = :log, xaxis=:log,
    linecolor = colours[1],
    # ylabel = "Error truncated radius",
    ylabel = "Root MSE for radius statistic",
    # ylabel = "Root MSE for truncated radius",
    xlabel = "Refreshment rate",
    # line = :solid,
    # linestyle=:dash,
    linewidth = 2,
    # title = title_plot,
    # legend = :topleft,
    # legendfontsize = 10,
    xticks = [0, 0.5, 1, 1.5, 2, 2.5, 3],
    # xticks = [0, 0.25, 0.5, 0.75, 1],
    # xticks = (δ,[0.975,0.95,0.925,0.9]),
    # ylims = [0, 1],
    # yticks=[10^0.5,10^0,10^(-0.5),10^(-1),10^(-1.5),10^(-2)],
)
q = plot!(
    refresh_rates,
    # avg_err[:, 2],
    sqrt.(avg_errors[:, 2]),
    label = "Splitting RDBDR",
    linewidth = 2,
    linecolor = colours[3],
)
q = plot!(
    refresh_rates,
    # avg_err[:, 3],
    sqrt.(avg_errors[:, 3]),
    label = "Splitting DRBRD",
    linewidth = 2,
    linecolor = colours[2],
)
q = plot!(
    refresh_rates,
    # avg_err[:, 4],
    sqrt.(avg_errors[:, 4]),
    label = "Splitting BDRDB",
    linestyle=:dash,
    linewidth = 2,
    linecolor = colours[4],
)

display(q)

# savefig(string("bps_empi_err_gauss_dim_",dim,"_rho_",ρ,"_delta_",δ,"_nexp_",n_exp,".pdf"))