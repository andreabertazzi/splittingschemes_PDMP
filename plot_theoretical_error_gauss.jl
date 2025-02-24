using Plots
using SpecialFunctions

p(x) = exp.(-(x.^2) / 2) / (sqrt(2 * pi))
p(x, v) = 0.5 * p(x)
λ_r = 3.0


## Plot the invariant distribution up to second order

# Splitting DBRBD
function f_2p_DBRBD(x::Real)
    (λ_r / 24) * (2 * sqrt(2 / pi) - x^3 * sign(x))
end
function f_2m_DBRBD(x::Real)
    f_2p_DBRBD(x)
end
function f_DBRBD(x::Real, v::Int64)
    f_2p_DBRBD(x)
end

function p_2_DBRBD(x::Real, v::Int64, δ::Real)
    return p(x) * (1- (δ^2) * f_DBRBD(x, v))
    # return p(x) * exp(-(δ^2) * f_DBRBD(x, v))
end
function p_2x_DBRBD(x::Real)
    return p(x) * (1- (δ^2) * f_DBRBD(x, v))
    p_2_DBRBD(x, v, δ)
end


# Splitting BDRBD
function f_2p_BDRDB(x::Real)
    D = 1 / 8
    if x <= 0
        return D - 1/4 * (x^2)
    else
        return D
    end
end
function f_2m_BDRDB(x::Real)
    D = 1 / 8
    if x <= 0
        return D
    else
        return D- 1/4 * (x^2)
    end
end
function f_BDRDB(x::Real)
    return 0.5*f_2p_BDRDB(x) + 0.5*f_2m_BDRDB(x)
end
function p_2_BDRDB(x::Real, δ::Real)
    return p(x) * (1- (δ^2) * f_BDRDB(x))
    # return p(x) * exp(-(δ^2) * f_BDRBD(x, v))
end
function p_2x_BDRDB(x::Real)
    p_2_BDRDB(x, δ)
end

# Splitting DRBRD
function f_2p_DRBRD(x::Real)
    D = (λ_r / 6) * sqrt(2 / pi) + (λ_r^2) / 16
    D - (λ_r / 12) * (x^3) * sign(x) - λ_r^2 * (x^2) / 16
end
function f_2m_DRBRD(x::Real)
    f_2p_DRBRD(x)
end
function f_DRBRD(x::Real, v::Int64)
    f_2p_DRBRD(x)
end
function p_2_DRBRD(x::Real, v::Int64, δ::Real)
    return p(x) * (1- (δ^2) * f_DRBRD(x, v))
    # return p(x) * exp(-(δ^2) * f_DRBRD(x, v))
end
function p_2x_DRBRD(x::Real)
    p_2_DRBRD(x, v, δ)
end

# h = 0.01
# pts = 3
# N = Integer(pts / h)
# x = [i * h for i = -N:N]
δ = 0.5
v = 1
colours = [:red,:orange,:green,:blue]
# plot(x, p_2x_DBRBD, label = "Splitting DBRBD",linewidth=2,linecolor=colours[1])
# plot!(x, p, label = "Splitting RDBDR",linewidth=2,linecolor=colours[2])
# plot!(x, p_2x_DRBRD, label = "Splitting DRBRD",linewidth=2,linecolor=colours[3])
# plot!(x, p_2x_BDRDB, label = "Splitting BDRDB",linewidth=2,linecolor=colours[4])
# plot!(x, p, label = "Ground truth",linecolor=:black,legendfontsize=10,linestyle=:dash)

# savefig(string("bps_invmeas_1d_refr_",λ_r,"_delta",δ,"new.pdf"))

# p
## Here estimate the TV distance
using Roots, QuadGK
# using HCubature

tol = 10^(-4)

diff_DBRBD(x) = p(x) - p_2x_DBRBD(x)
roots_DBRBD = find_zeros(diff_DBRBD,-8,8)
A = abs(roots_DBRBD[1])
integral1, err = quadgk(x -> (p_2x_DBRBD(x)-p(x)), A, Inf, rtol=1e-8)
integral2, err = quadgk(x -> (p_2x_DBRBD(x)-p(x)), -Inf,-A, rtol=1e-8)
out_DBRBD = abs(integral1)+abs(integral2)
in_DBRBD,err = quadgk(x -> (p(x)-p_2x_DBRBD(x)), -A, A, rtol=1e-8)
#everything is symmetric, so the difference is equal with opposite sign!

diff_BDRDB(x) = p(x) - p_2x_BDRDB(x)
roots_BDRDB = find_zeros(diff_BDRDB,-8,8)
A = abs(roots_BDRDB[1])
integral1, err = quadgk(x -> (p_2x_BDRDB(x)-p(x)), A, Inf, rtol=1e-8)
integral2, err = quadgk(x -> (p_2x_BDRDB(x)-p(x)), -Inf,-A, rtol=1e-8)
out_BDRDB = abs(integral1)+abs(integral2)
in_BDRDB,err = quadgk(x -> (p(x)-p_2x_BDRDB(x)), -A, A, rtol=1e-8)

diff_DRBRD(x) = p(x) - p_2x_DRBRD(x)
roots_DRBRD = find_zeros(diff_DRBRD,-8,8)
A = abs(roots_DRBRD[1])
integral1, err = quadgk(x -> (p_2x_DRBRD(x)-p(x)), A, Inf, rtol=1e-8)
integral2, err = quadgk(x -> (p_2x_DRBRD(x)-p(x)), -Inf,-A, rtol=1e-8)
out_DRBRD = abs(integral1)+abs(integral2)
in_DRBRD,err = quadgk(x -> (p(x)-p_2x_DRBRD(x)), -A, A, rtol=1e-8)

## Check that terms integrate to 0

# f_sum_DBRBD(x,λ_r) =  2 * (λ_r / 24) * (2 * sqrt(2 / pi) - x^3 * sign(x))
# quadgk(x -> (f_sum_DBRBD(x,1)*p(x)), -Inf, Inf, rtol=1e-8)
#
# f_sum_BDRDB(x) = 1/4 * (1 - x^2)
# quadgk(x -> (f_sum_BDRDB(x)*p(x)), -Inf, Inf, rtol=1e-8)
#
# D_DRBRD(λ_r) = (λ_r / 6) * sqrt(2 / pi) + (λ_r^2) / 16
# f_sum_DRBRD(x,λ_r) = 2*(D_DRBRD(λ_r) - (λ_r / 12) * (x^3) * sign(x) - λ_r^2 * (x^2) / 16)
# quadgk(x -> (f_sum_DRBRD(x,3)*p(x)), -Inf, Inf, rtol=1e-8)


## Plot TV distance as function of refreshment rate
function plot_as_fn_refreshrate()
    stepsize = 0.05
    λs = [i*stepsize for i=1:60]
    DBRBD = Vector{Float64}(undef, length(λs))
    BDRDB = Vector{Float64}(undef, length(λs))
    DRBRD = Vector{Float64}(undef, length(λs))
    for (k,rate) in enumerate(λs)
        λ_r = rate
        # diff_DBRBD(x) = p(x) - p_2x_DBRBD(x)
        f_sum_DBRBD(x) = 2 * (λ_r / 24) * (2 * sqrt(2 / pi) - x^3 * sign(x))
        roots_DBRBD = find_zeros(f_sum_DBRBD,-8,8)
        # diff_BDRDB(x) = p(x) - p_2x_BDRDB(x)
        f_sum_BDRDB(x) = 1/4 * (1 - x^2)
        roots_BDRDB = find_zeros(f_sum_BDRDB,-8,8)
        # diff_DRBRD(x) = p(x) - p_2x_DRBRD(x)
        D_DRBRD = (λ_r / 6) * sqrt(2 / pi) + (λ_r^2) / 16
        f_sum_DRBRD(x) = 2*(D_DRBRD - (λ_r / 12) * (x^3) * sign(x) - λ_r^2 * (x^2) / 16)
        roots_DRBRD = find_zeros(f_sum_DRBRD,-8,8)
        A = abs(roots_DBRBD[1])
        DBRBD[k], err = quadgk(x -> (f_sum_DBRBD(x)*p(x)), -A, A, rtol=1e-8)
        A = abs(roots_BDRDB[1])
        BDRDB[k], err = quadgk(x -> (f_sum_BDRDB(x)*p(x)), -A, A, rtol=1e-8)
        A = abs(roots_DRBRD[1])
        DRBRD[k], err = quadgk(x -> (f_sum_DRBRD(x)*p(x)), -A, A, rtol=1e-8)
    end
    (DBRBD,BDRDB,DRBRD)
end

(DBRBD,BDRDB,DRBRD) = plot_as_fn_refreshrate()
λs = [i*0.05 for i=1:60]
stepsize = 0.5
DBRBD = 0.5*(stepsize^2)*DBRBD
prepend!(DBRBD,0.)
BDRDB = 0.5*(stepsize^2)*BDRDB
prepend!(BDRDB,BDRDB[1])
DRBRD = 0.5*(stepsize^2)*DRBRD
prepend!(DRBRD,0.)
prepend!(λs,0.)
RDBDR = zeros(length(λs))
#
colours = [:red,:orange,:green,:blue]
plot(λs,DBRBD,ylabel = "TV distance", xlabel = "Refreshment rate",
    linewidth=2,legend=:topleft,legendfontsize=10,
    linecolor=colours[1], label = "Splitting DBRBD",
    xticks=[0,0.5,1,1.5,2,2.5,3])
plot!(λs,RDBDR,linewidth=2,linecolor=colours[2], label = "Splitting RDBDR",)
plot!(λs,DRBRD,linewidth=2,linecolor=colours[3], label = "Splitting DRBRD",)
plot!(λs,BDRDB,linewidth=2,linecolor=colours[4], label = "Splitting BDRDB",)
# savefig(string("bps_tvdistance_1d_delta_",stepsize,".pdf"))

## Hoping for some ordering given by the h function
# using QuadGK # to compute integrals
# function plot_h_as_fn_refreshrate()
#     stepsize = 0.05
#     λs = [i*stepsize for i=1:60]
#     DBRBD = Vector{Float64}(undef, length(λs))
#     BDRDB = Vector{Float64}(undef, length(λs))
#     DRBRD = Vector{Float64}(undef, length(λs))
#     for k = 1:length(λs)
#         λ_r = λs[k]
#         # diff_DBRBD(x) = p(x) - p_2x_DBRBD(x)
#         f_sum_DBRBD(x) = 2 * (λ_r / 24) * (2 * sqrt(2 / pi) - x^3 * sign(x))
#         roots_DBRBD = find_zeros(f_sum_DBRBD,-8,8)
#         # diff_BDRDB(x) = p(x) - p_2x_BDRDB(x)
#         f_sum_BDRDB(x) = 1/4 * (1 - x^2)
#         roots_BDRDB = find_zeros(f_sum_BDRDB,-8,8)
#         # diff_DRBRD(x) = p(x) - p_2x_DRBRD(x)
#         D_DRBRD = (λ_r / 6) * sqrt(2 / pi) + (λ_r^2) / 16
#         f_sum_DRBRD(x) = 2*(D_DRBRD - (λ_r / 12) * (x^3) * sign(x) - λ_r^2 * (x^2) / 16)
#         roots_DRBRD = find_zeros(f_sum_DRBRD,-8,8)
#         A = abs(roots_DBRBD[1])
#         DBRBD[k], err = quadgk(x -> (f_sum_DBRBD(x)*p(x)), -A, A, rtol=1e-8)
#         A = abs(roots_BDRDB[1])
#         BDRDB[k], err = quadgk(x -> (f_sum_BDRDB(x)*p(x)), -A, A, rtol=1e-8)
#         A = abs(roots_DRBRD[1])
#         DRBRD[k], err = quadgk(x -> (f_sum_DRBRD(x)*p(x)), -A, A, rtol=1e-8)
#     end
#     (DBRBD,BDRDB,DRBRD)
# end
#
# h = 0.01
# pts = 5
# N = Integer(pts / h)
# x = [i * h for i = -N:N]
#
# h_DBRBD(x,v,λ) = (λ/8) * (x^2 + 2*v*x*max(0,(-v*x)))
# h_BDRDB(x,v,λ) = (1/8) * (-λ*(x^2 + 2*v*x*max(0,(-v*x))) + 2*max(-(v*x),0)*((x^2)-2))
# h_DRBRD(x,v,λ) = (λ/8) * (x^2 + v*x*(3*max(0,(-v*x))+max(0,(v*x))) +λ*v*x )
#
# λ_r = 1.0
# hs_DBRBD(x) = abs(h_DBRBD(x,1,λ_r)) + abs(h_DBRBD(x,-1,λ_r))
# hs_BDRDB(x) = abs(h_BDRDB(x,1,λ_r)) + abs(h_BDRDB(x,-1,λ_r))
# hs_DRBRD(x) = abs(h_DRBRD(x,1,λ_r)) + abs(h_DRBRD(x,-1,λ_r))
# plot(x,hs_DBRBD)
# plot!(x,hs_BDRDB)
# plot!(x,hs_DRBRD)
#
# hw_DBRBD(x) = hs_DBRBD(x) * p(x)
# hw_BDRDB(x) = hs_BDRDB(x) * p(x)
# hw_DRBRD(x) = hs_DRBRD(x) * p(x)
# plot(x,hw_DBRBD)
# plot!(x,hw_BDRDB)
# plot!(x,hw_DRBRD)
# integral_DBRBD, err = quadgk(x -> hw_DBRBD(x), -Inf, Inf, rtol=1e-8)
# integral_BDRDB, err = quadgk(x -> hw_BDRDB(x), -Inf, Inf, rtol=1e-8)
# integral_DRBRD, err = quadgk(x -> hw_DRBRD(x), -Inf, Inf, rtol=1e-8)

## Checks
# integral1, err = quadgk(x -> p_2x_DRBRD(x), -Inf, Inf, rtol=1e-8)
# fun1(x) = abs(p(x)-p_2x_DBRBD(x))
# fun2(x) = abs(p(x)-p_2x_BDRDB(x))
# integral1, err = quadgk(x -> fun(x), -Inf, Inf, rtol=1e-8)
# plot(x,fun1)
# plot!(x,fun2)
