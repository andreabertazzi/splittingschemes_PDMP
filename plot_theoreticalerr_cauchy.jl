include("helper_split.jl")

using Plots
using SpecialFunctions
using QuadGK, Roots

p(x) = 1/(pi*(1+x^2))
p(x, v) = 0.5 * p(x)

# DBRBD
f0_DBRBD(λ) =  ( (λ / 4) * (pi/4 -1/pi) -1/16)
f_DBRBD(x,λ) = f0_DBRBD(λ) - λ/4 * (abs(atan(x))-abs(x)/(1+x^2)) - 1/12*((1-x^2)/((1+x^2)^2) -1)
f0_DBRBD(λ) =   (λ / 4) * (pi/4 -1/pi) +(1/48)
f_DBRBD(x,λ) = f0_DBRBD(λ) - λ/4 * (abs(atan(x))-abs(x)/(1+x^2)) - 1/12*(1-x^2)/((1+x^2)^2)
# stepsize = 0.05
# xs = [i*stepsize for i=-60:60]
# plot(xs,f_DBRBD.(xs,1))
# quadgk(x -> (f_DBRBD(x,2)*p(x)), -Inf, Inf, rtol=1e-8)
# plot(xs,f_DBRBD.(xs,.2))

# RDBDR
f_RDBDR(x) = 1/12 *(1/4 + (x^2-1)/((1+x^2)^2))
# quadgk(x -> (f_RDBDR(x)*p(x)), -Inf, Inf, rtol=1e-8)
# plot(xs,f_RDBDR.(xs))

# BDRDB
f_sum_BDRDB(x) = ((x^2-3)^2)/(48*(x^2+1)^2) + (x^4-54*x^2+9)/(48*(x^2+1)^2)
# quadgk(x -> (f_sum_BDRDB(x)*p(x)), -Inf, Inf, rtol=1e-8)
# stepsize = 0.05
# xs = [i*stepsize for i=-1000:1000]
# plot(xs,f_sum_BDRDB.(xs,1))


# DRBRD
term1_DRBRD(x,λ) = 0.5 * λ * ( pi/4 - abs(atan(x)) +abs(x)/(1+x^2) -1/pi)
term2_DRBRD(x,λ) = 1/12 *(1/4 + (x^2-1)/((1+x^2)^2))
term3_DRBRD(x,λ) = ((λ^2)/8) * (log(4/(1+x^2)))
f_DRBRD(x,λ) = term1_DRBRD(x,λ) + term2_DRBRD(x,λ) + term3_DRBRD(x,λ)
quadgk(x -> (f_DRBRD(x,0.5)*p(x)), -100000000000, 100000000000, rtol=1e-8)

# stepsize = 0.5
# λs = [i*0.05 for i=0:60]
# bias0_DBRBD(λ) = f_DBRBD(0,λ)   * (stepsize^2) * p(0)
# bias0_RDBDR(λ) = f_RDBDR(0)     * (stepsize^2) * p(0)
# bias0_DRBRD(λ) = f_DRBRD(0,λ)   * (stepsize^2) * p(0)
# bias0_BDRDB(λ) = f_sum_BDRDB(0) * (stepsize^2) * p(0) / 2

# bias0_BDRDB(λ) = f_sum_BDRDB(0,λ) * (stepsize^2) * p(0) / 2

# colours = [:red,:orange,:green,:blue]
# plot(λs,bias0_DBRBD,ylabel = "Bias at x=0", xlabel = "Refreshment rate",
#     linewidth=2,legend=:topleft,legendfontsize=10,
#     linecolor=colours[1], label = "Splitting DBRBD",
#     xticks=[0,0.5,1,1.5,2,2.5,3],
#     # ylim = [0.01,0.2]
#     )
# plot!(λs,bias0_RDBDR,linewidth=2,linecolor=colours[2], label = "Splitting RDBDR",)
# plot!(λs,bias0_DRBRD,linewidth=2,linecolor=colours[3], label = "Splitting DRBRD",)
# plot!(λs,bias0_BDRDB,linewidth=2,linecolor=colours[4], label = "Splitting BDRDB",)
# savefig(string("bps_biasat0_cauchy_1d_delta_",stepsize,"_new.pdf"))


##
function plot_as_fn_refreshrate()
    DBRBD = Vector{Float64}(undef, length(λs))
    # BDRDB = Vector{Float64}(undef, length(λs))
    DRBRD = Vector{Float64}(undef, length(λs))
    # RDBDR = Vector{Float64}(undef, length(λs))

    f0_DBRBD(λ) =  ( (λ / 4) * (pi/4 -1/pi) -1/16)
    f0_DBRBD(x,λ) = ( f0_DBRBD(λ) - λ/4 * (abs(atan(x))-abs(x)/(1+x^2)) - 1/12*((1-x^2)/((1+x^2)^2) -1))

    term1_DRBRD(x,λ) = 0.5 * λ * ( pi/4 - abs(atan(x)) +abs(x)/(1+x^2) -1/pi)
    term2_DRBRD(x,λ) = 1/12 *(1/4 + (x^2-1)/((1+x^2)^2))
    term3_DRBRD(x,λ) = ((λ^2)/8) * (log(4/(1+x^2)))
    f_DRBRD(x,λ) = term1_DRBRD(x,λ) + term2_DRBRD(x,λ) + term3_DRBRD(x,λ)

    for k = 1:length(λs)
        f_sum_DBRBD(x) = 2 * f_DBRBD(x,λs[k])
        roots_DBRBD = find_zeros(f_sum_DBRBD,-20,20)
        DBRBD[k] = compute_tv_dist(f_sum_DBRBD, roots_DBRBD, p)
        DBRBD[k] = 0.5*(stepsize^2)*DBRBD[k]

        f_sum_DRBRD(x) = 2 * f_DRBRD(x,λs[k])
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
f_sum_RDBDR(x) = 2 * f_RDBDR(x)
roots_RDBDR = find_zeros(f_sum_RDBDR,-10,10)
val_RDBDR = compute_tv_dist(f_sum_RDBDR, roots_RDBDR, p)
RDBDR= 0.5*(stepsize^2)*val_RDBDR * ones(length(λs))

f_sum_BDRDB(x) = ((x^2-3)^2)/(48*(x^2+1)^2) + (x^4-54*x^2+9)/(48*(x^2+1)^2)
roots_BDRDB = find_zeros(f_sum_BDRDB,-10,10)
val_BDRDB = compute_tv_dist(f_sum_BDRDB, roots_BDRDB, p)
BDRDB = 0.5 * (stepsize^2) * val_BDRDB * ones(length(λs))


colours = [:red,:orange,:green,:blue]
plot(λs,DBRBD,ylabel = "TV distance", xlabel = "Refreshment rate",
    linewidth=2,legend=:topleft,legendfontsize=10,
    linecolor=colours[1], label = "Splitting DBRBD",
    xticks=[0,0.5,1,1.5,2,2.5,3],
    ylim = [-0.01,0.24]
    )
plot!(λs,RDBDR,linewidth=2,linecolor=colours[2], label = "Splitting RDBDR",)
plot!(λs,DRBRD,linewidth=2,linecolor=colours[3], label = "Splitting DRBRD",)
plot!(λs,BDRDB,linewidth=2,linecolor=colours[4], label = "Splitting BDRDB",)

# savefig(string("bps_tvdistance_cauchy_1d_delta_",stepsize,".pdf"))

##
# function find_bias_BDRDB_cauchy(rr::Vector{Float64})
#     f_plus_BDRDB(x,λ) = (1/6)*((1-x^2)/((1+x^2)^2)-1) -(1/4) * indicator_fn(x, -Inf, 0)*( 1-(1+4*x^2)/((1+x^2)^2) )
#     g_BDRDB(x,λ) = -x^2 * sign(x) / ((1+x^2)^2)
#     f_minus_BDRDB(x,λ) = f_plus_BDRDB(x,λ) + g_BDRDB(x,λ)
#     f_sum_BDRDB(x,λ) = f_plus_BDRDB(x,λ) + f_minus_BDRDB(x,λ)
#     bias_0 = Vector{Float64}(undef,length(rr))
#     for i = 1 : length(rr)
#         f_2_0 = quadgk(x -> (-f_sum_BDRDB(x,rr[i])*p(x)), -Inf, Inf, rtol=1e-8)
#         f_BDRDB(x,i) = f_2_0[1] + f_sum_BDRDB(x,rr[i])
#         bias_0[i] = f_BDRDB(0,i)  * (stepsize^2) * p(0) / 2
#     end
#     bias_0
# end
