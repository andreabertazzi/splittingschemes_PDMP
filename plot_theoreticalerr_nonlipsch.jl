include("helper_split.jl")

using Plots
using SpecialFunctions
using QuadGK, Roots

fn(x) = exp(-x^4)
partition_fn, err = quadgk(x -> fn(x), -Inf, Inf, rtol=1e-8)
p(x) = fn(x)/partition_fn
p(x, v) = 0.5 * p(x)

##

function plot_as_fn_refreshrate()
    DBRBD = Vector{Float64}(undef, length(λs))
    # BDRDB = Vector{Float64}(undef, length(λs))
    DRBRD = Vector{Float64}(undef, length(λs))
    # RDBDR = Vector{Float64}(undef, length(λs))

    for k = 1:length(λs)
        f_sum_DBRBD(x) = 2 * ((λs[k]/ 7) * (1/(2*gamma(5/4)) - 2*x^7 * sign(x)) + 0.5*(gamma(3/4)/gamma(1/4) - x^2))
        roots_DBRBD = find_zeros(f_sum_DBRBD,-20,20)
        DBRBD[k] = compute_tv_dist(f_sum_DBRBD, roots_DBRBD, p)
        DBRBD[k] = 0.5*(stepsize^2)*DBRBD[k]

        f_sum_DRBRD(x) = 2*( (λs[k] / 7) * (1/(gamma(5/4)) - 4*x^7 * sign(x)) + 0.5*(gamma(3/4)/gamma(1/4) - x^2) + (λs[k]^2)/8 * (0.25-x^4))
        roots_DRBRD = find_zeros(f_sum_DRBRD,-20,20)
        DRBRD[k] = compute_tv_dist(f_sum_DRBRD, roots_DRBRD, p)
        DRBRD[k] = 0.5*(stepsize^2)*DRBRD[k]
    end
    # (DBRBD,BDRDB,DRBRD,RDBDR)
    (DBRBD,DRBRD)
end

stepsize = 0.5
λs = [i*0.05 for i=0:60]
(DBRBD,DRBRD) = plot_as_fn_refreshrate()

f_RDBDR(x) = 1/12 *(1/4 + (x^2-1)/((1+x^2)^2))
f_sum_RDBDR(x) = 2*( gamma(3/4)/(8*gamma(5/4)) - x^2/2)
roots_RDBDR = find_zeros(f_sum_RDBDR,-10,10)
val_RDBDR = compute_tv_dist(f_sum_RDBDR, roots_RDBDR, p)
RDBDR= 0.5*(stepsize^2)*val_RDBDR * ones(length(λs))

f_sum_BDRDB(x) = 5*gamma(3/4)/gamma(1/4) - 4*x^6- 2*x^2
roots_BDRDB = find_zeros(f_sum_BDRDB,-10,10)
val_BDRDB = compute_tv_dist(f_sum_BDRDB, roots_BDRDB, p)
BDRDB = 0.5 * (stepsize^2) * val_BDRDB * ones(length(λs))

colours = [:red,:orange,:green,:blue]
plot(λs,DBRBD,ylabel = "TV distance", xlabel = "Refreshment rate",
    linewidth=2,legend=:topleft,legendfontsize=10,
    linecolor=colours[1], label = "Splitting DBRBD",
    xticks=[0,0.5,1,1.5,2,2.5,3],
    ylim = [0,0.2])
plot!(λs,RDBDR,linewidth=2,linecolor=colours[2], label = "Splitting RDBDR",)
plot!(λs,DRBRD,linewidth=2,linecolor=colours[3], label = "Splitting DRBRD",)
plot!(λs,BDRDB,linewidth=2,linecolor=colours[4], label = "Splitting BDRDB",)

# savefig(string("bps_tvdistance_nonlipsch_1d_delta_",stepsize,"new.pdf"))


# f_sum_DBRBD(x,λ_r) = 2 * ((λ_r / 7) * (1/(2*gamma(5/4)) - 2*x^7 * sign(x)) + 0.5*(gamma(3/4)/gamma(1/4) - x^2))
# quadgk(x -> (f_sum_DBRBD(x,4)*p(x)), -Inf, Inf, rtol=1e-8)
#
# f_sum_BDRDB(x) = 5*gamma(3/4)/gamma(1/4) - 4*x^6- 2*x^2
# quadgk(x -> (f_sum_BDRDB(x)*p(x)), -Inf, Inf, rtol=1e-8)
#
# f_sum_DRBRD(x,λ_r) = 2*( (λ_r / 7) * (1/(gamma(5/4)) - 4*x^7 * sign(x)) + 0.5*(gamma(3/4)/gamma(1/4) - x^2) + (λ_r^2)/8 * (0.25-x^4))
# quadgk(x -> (f_sum_DRBRD(x,0)*p(x)), -Inf, Inf, rtol=1e-8)
#
#
# f_sum_RDBDR(x) = 2*( gamma(3/4)/(8*gamma(5/4)) - x^2/2)
# quadgk(x -> (f_sum_RDBDR(x)*p(x)), -Inf, Inf, rtol=1e-8)


# function plot_as_fn_refreshrate()
#     stepsize = 0.05
#     λs = [i*stepsize for i=0:60]
#     DBRBD = Vector{Float64}(undef, length(λs))
#     BDRDB = Vector{Float64}(undef, length(λs))
#     DRBRD = Vector{Float64}(undef, length(λs))
#     RDBDR = Vector{Float64}(undef, length(λs))
#     for k = 1:length(λs)
#         λ_r = λs[k]
#         # diff_DBRBD(x) = p(x) - p_2x_DBRBD(x)
#         f_sum_DBRBD(x) = 2 * ((λ_r / 7) * (1/(2*gamma(5/4)) - 2*x^7 * sign(x)) + 0.5*(gamma(3/4)/gamma(1/4) - x^2))
#         roots_DBRBD = find_zeros(f_sum_DBRBD,-10,10)
#         # diff_BDRDB(x) = p(x) - p_2x_BDRDB(x)
#         f_sum_BDRDB(x) = 5*gamma(3/4)/gamma(1/4) - 4*x^6- 2*x^2
#         #(gamma(7/4)/gamma(5/4) - 4*x^6) + 2*(gamma(3/4)/gamma(1/4) - x^2)
#         roots_BDRDB = find_zeros(f_sum_BDRDB,-10,10)
#         # diff_DRBRD(x) = p(x) - p_2x_DRBRD(x)
#         f_sum_DRBRD(x) = 2*( (λ_r / 7) * (1/(gamma(5/4)) - 4*x^7 * sign(x)) + 0.5*(gamma(3/4)/gamma(1/4) - x^2) + (λ_r^2)/8 * (0.25-x^4))
#         roots_DRBRD = find_zeros(f_sum_DRBRD,-10,10)
#         f_sum_RDBDR(x) = 2*( gamma(3/4)/(8*gamma(5/4)) - x^2/2)
#         roots_RDBDR = find_zeros(f_sum_RDBDR,-10,10)
#         A = abs(roots_DBRBD[1])
#         val_1, err = quadgk(x -> (f_sum_DBRBD(x)*p(x)), -A, A, rtol=1e-8)
#         val_2, err = quadgk(x -> (f_sum_DBRBD(x)*p(x)), -Inf, A, rtol=1e-8)
#         val_2 *= 2
#         DBRBD[k] = max(val_1, val_2)
#         A = abs(roots_BDRDB[1])
#         val_1, err = quadgk(x -> (f_sum_BDRDB(x)*p(x)), -A, A, rtol=1e-8)
#         val_2, err = quadgk(x -> (f_sum_BDRDB(x)*p(x)), -Inf, A, rtol=1e-8)
#         val_2 *= 2
#         BDRDB[k] = max(val_1, val_2)
#         A = abs(roots_DRBRD[1])
#         val_1, err = quadgk(x -> (f_sum_DRBRD(x)*p(x)), -A, A, rtol=1e-8)
#         val_2, err = quadgk(x -> (f_sum_DRBRD(x)*p(x)), -Inf, A, rtol=1e-8)
#         val_2 *= 2
#         DRBRD[k] = max(val_1, val_2)
#         A = abs(roots_RDBDR[1])
#         val_1, err = quadgk(x -> (f_sum_RDBDR(x)*p(x)), -A, A, rtol=1e-8)
#         val_2, err = quadgk(x -> (f_sum_RDBDR(x)*p(x)), -Inf, A, rtol=1e-8)
#         val_2 *= 2
#         RDBDR[k] = max(val_1, val_2)
#     end
#     (DBRBD,BDRDB,DRBRD,RDBDR)
# end
#
# (DBRBD,BDRDB,DRBRD,RDBDR) = plot_as_fn_refreshrate()
# λs = [i*0.05 for i=0:60]
# stepsize = 0.5
# DBRBD = 0.5*(stepsize^2)*DBRBD
# BDRDB = 0.5*(stepsize^2)*BDRDB
# DRBRD = 0.5*(stepsize^2)*DRBRD
# RDBDR = 0.5*(stepsize^2)*RDBDR
# #
# colours = [:red,:orange,:green,:blue]
# plot(λs,DBRBD,ylabel = "TV distance", xlabel = "Refreshment rate",
#     linewidth=2,legend=:topleft,legendfontsize=10,
#     linecolor=colours[1], label = "Splitting DBRBD",
#     xticks=[0,0.5,1,1.5,2,2.5,3],
#     ylim = [0,0.2])
# plot!(λs,RDBDR,linewidth=2,linecolor=colours[2], label = "Splitting RDBDR",)
# plot!(λs,DRBRD,linewidth=2,linecolor=colours[3], label = "Splitting DRBRD",)
# plot!(λs,BDRDB,linewidth=2,linecolor=colours[4], label = "Splitting BDRDB",)

# savefig(string("bps_tvdistance_nonlipsch_1d_delta_",stepsize,"new.pdf"))
