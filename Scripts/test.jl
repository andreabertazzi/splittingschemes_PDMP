include(joinpath(@__DIR__, "..","functions_particles.jl"))
using Distributions
N = 25 # number of particles

# Choose potential functions V and W and their derivatives
V(r) = r^4
W(r) = - sqrt(1 + r^2)
Vprime(r) = 4 * (r^3)
Wprime(r) = r / W(r)

# Define potential and its gradient
function interaction_pot_grad(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + sum(Wprime.(x .- x'), dims=2) / (length(x)))
end
function interaction_pot_grad_stoch(x::Vector{Float64},j::Integer) 
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])) + Wprime.(x .- x[j]) )
end
function gradV(x::Vector{Float64})
    return vec(vcat(Vprime.(x[1:end-1] - x[2:end]), 0.0) - vcat(0.0, Vprime.(x[1:end-1] - x[2:end])))
end
function gradW(x::Vector{Float64},j::Integer)
    return vec(Wprime.(x .- x[j]))
end
function interaction_pot(x::Vector{Float64})
    return sum(V.(x[1:end-1] - x[2:end])) + sum(W.(x .- x')) / (2 * length(x))
end
function define_gradVi(Vprime::Function, x::Vector{<:Real}, i::Integer, N::Integer)
    if i == 1
      return Vprime(x[1]-x[2])
    elseif i < N
        return ( Vprime(x[i]-x[i+1]) - Vprime(x[i-1]-x[i]) )
    elseif i==N
      return  - Vprime(x[N-1]-x[N])
    else
      error("Index larger than number of particles")
    end
end
gradVi(x,i) = define_gradVi(Vprime, x, i, N)

# Define the test function 
fun_var(x) = mean((x.-mean(x)).^2)

## Initial conditions
sigma = 1.
initial_state_ZZ() = initial_position_particles(N,sigma), rand((-1,1),N)
x,v = initial_state_ZZ() 
estimate = (2328.9060416376865, 2737.2679824910224) # for N=100, with BPS 16hrs compute time
ub = 1.
Wrates(x,v,i,j) = define_Wrates(Wprime, x, v, i, j)
# timesvec_1 = draw_exponential_time.(v .* vgrads)
# println("$v")
vgrads = gradV(x) .* v;
# @btime jump_part_particles_seq!(vgrads, Wrates, 1., x,v, 1e-2, N)
# @btime jump_part_particles!(vgrads, Wrates, 1., x,v, 1e-2, N)
# @btime jump_part_particles!(vgrads, Wrates, 1., x,v, 1e-2, n)
rt1 = @elapsed(jump_part_particles!(vgrads, Wrates, 1., x,v, 1e-2, N))
rt2 = @elapsed(jump_part_particles_seq!(vgrads, Wrates, 1., x,v, 1e-2, N))
println("Time for jump_part_particles! is $rt1, time for jump_part_particles_seq! is $rt2")
# println("$v")

n_iterations = 5 * 10^4   
n_updates = 5 * 10^0
K = 3
M = 3
n_obs = Int(ceil(n_iterations/n_updates))
# rt_zz = @elapsed(splitting_zzs_particles(gradV,Wrates,fun_var,ub,1e-2,N,n_iterations,n_updates,x,v))
# rt_zz = @elapsed(splitting_zzs_particles(gradVi,Wrates,fun_var,ub,1e-2,N,n_iterations,n_updates,x,v))
# println("Time for ZZS is $rt_zz")
# initial_state_HMC_laplace() = initial_position_particles(N,sigma), draw_laplace(N)
# x,v = initial_state_HMC_laplace();
# rt_hmc = @elapsed SHMC_Laplace(gradV,gradW,fun_var,N,1e-2,K,M,n_iterations,n_updates,x,v)

## try the multinomial approach without saving anything!
## avoid the gradV eval just before the function and just give the function directly, This avoids storing the vector