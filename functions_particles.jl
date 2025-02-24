include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")

# function pos_part(x::Real)
#     max(x,0)
# end

# function define_Vrates(Vprime::Function, x::Vector{<:Real}, v::Vector{<:Real}, i::Integer, N::Integer)
#     if i == 1
#       return pos_part(v[1] * Vprime(x[1]-x[2]))
#     elseif i < N
#         return pos_part(v[i] * ( Vprime(x[i]-x[i+1]) - Vprime(x[i-1]-x[i]) ))
#     elseif i==N
#       return pos_part(-v[N] * Vprime(x[N-1]-x[N]))
#     else
#       error("Index larger than number of particles")
#     end
# end

# function define_Vgrad_ZZ(Vprime::Function, x::Vector{<:Real}, v::Vector{<:Real}, i::Integer, N::Integer)
#     if i == 1
#       return Vprime(x[1]-x[2])
#     elseif i < N
#         return ( Vprime(x[i]-x[i+1]) - Vprime(x[i-1]-x[i]) )
#     elseif i==N
#       return (- Vprime(x[N-1]-x[N]))
#     else
#       error("Index larger than number of particles")
#     end
# end
  
function define_Wrates(Wprime::Function, x::Vector{<:Real}, v::Vector{<:Real}, i::Integer, j::Integer)
    # max( 0, v[i] * Wprime(x[i]-x[j]) ) 
    v[i] * Wprime(x[i]-x[j]) 
end

function define_Vgrad(Vprime::Function, x::Vector{<:Real}, N::Integer)
    grad = Vector{Float64}(undef,N)
    grad[1] = Vprime(x[1]-x[2]) 
    for i = 2:N-1
        grad[i] = Vprime(x[i]-x[i+1]) - Vprime(x[i-1]-x[i])
    end
    grad[N] = Vprime(x[N-1]-x[N])
    grad
end

function define_Wgrad(Wprime::Function, x::Vector{<:Real}, j::Integer, N::Integer)
    grad = Vector{Float64}(undef,N)
    for i = 1:N
        grad[i] = Wprime(x[i]-x[j])
    end
    grad
end

function initial_position_particles(N::Integer,sigma::Real)
    x_init = sigma * randn(N)
    # sort!(x_init)
    x_init
end

function initial_state_zzs_particles(N::Integer, sigma::Real)
    x_init = initial_position_particles(N,sigma)
    v_init = rand((-1,1),N)
    x_init,v_init
end

function initial_velocity_sphere(dim::Integer)
    t = randn(dim)
    t / norm(t)
end

function draw_laplace(dim::Integer)
    u = rand(dim)
    v = - sign.(u .- 0.5) .* log.(1 .- 2 * abs.(u .- 0.5))
    v
end


## Algorithms 

function splitting_zzs_particles(
                                Vgrad::Function,
                                Wrates::Function,
                                func::Function,
                                ub_W::Real,
                                δ::Float64,
                                N::Integer,
                                n_evals::Integer,
                                x_init::Vector{<:Real},
                                v_init::Vector{Int64})

    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    v = copy(v_init)
    n_grads = 0 
    n_iter = 0
    while n_grads < N * n_evals
        n_iter += 1
        x += v * δ/2
        Vgrad_eval = Vgrad(x)
        n_grads += jump_part_particles!(Vgrad_eval, Wrates, ub_W, x, v, δ, N)
        x += v * δ/2
        eval_x = func(x)
        old_est_mean = est_mean
        est_mean += (1/n_iter) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
    end

    est_mean, est_var/n_iter

end

# function splitting_zzs_particles(
#                                 Vgrad::Function,
#                                 Wrates::Function,
#                                 func::Function,
#                                 ub_W::Real,
#                                 δ::Float64,
#                                 N::Integer,
#                                 n_iters::Integer,
#                                 n_btw::Integer,
#                                 x_init::Vector{<:Real},
#                                 v_init::Vector{Int64})

#     est_mean = 0.
#     est_var = 0.
#     estims_vec = Vector{Tuple{Float64,Float64}}()
#     x = copy(x_init)
#     v = copy(v_init)
#     n_outerloop = Int(ceil(n_iters / n_btw))
#     n_iter = 0
#     Vgrad_eval = Vector{Float64}(undef,N)
#     for i in 1:n_outerloop
#         for j in 1:n_btw
#             n_iter +=1
#             x += v * δ/2
#             Vgrad_eval[:] = Vgrad(x)
#             # jump_part_particles_seq!(Vgrad_eval, Wrates, ub_W, x, v, δ, N)
#             jump_part_particles!(Vgrad_eval, Wrates, ub_W, x, v, δ, N)
#             x += v * δ/2
#             eval_x = func(x)
#             old_est_mean = est_mean
#             est_mean += (1/n_iter) * ( eval_x - est_mean )
#             est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
#         end
#         push!(estims_vec,(est_mean,est_var/n_iter))
#     end

#     estims_vec

# end

function splitting_zzs_particles(
                                Vgrad::Function,
                                Wrates::Function,
                                func::Function,
                                ub_W::Real,
                                δ::Float64,
                                N::Integer,
                                n_iters::Integer,
                                n_btw::Integer,
                                x_init::Vector{<:Real},
                                v_init::Vector{Int64})

    est_mean = 0.
    est_var = 0.
    estims_vec = Vector{Tuple{Float64,Float64}}()
    x = copy(x_init)
    v = copy(v_init)
    n_outerloop = Int(ceil(n_iters / n_btw))
    n_iter = 0
    Vgrad_eval = Vector{Float64}(undef,N)
    for i in 1:n_outerloop
        for j in 1:n_btw
            n_iter +=1
            x += v * δ/2
            Vgrad_eval[:] = Vgrad(x)
            # jump_part_particles_seq!(Vgrad_eval, Wrates, ub_W, x, v, δ, N)
            jump_part_particles!(Vgrad_eval, Wrates, ub_W, x, v, δ, N)
            x += v * δ/2
            eval_x = func(x)
            old_est_mean = est_mean
            est_mean += (1/n_iter) * ( eval_x - est_mean )
            est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        end
        push!(estims_vec,(est_mean,est_var/n_iter))
    end

    estims_vec

end


function splitting_bps_particles(
    Vgrad::Function, # gradient for the V part of the potential
    Wgrad::Function, # gradient for the W part of the potential
    func::Function,
    ub_W::Real,
    δ::Float64,
    N::Integer,
    n_evals::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{<:Real},
    refr_rate::Real = 1.;
    vel_on_sphere::Bool = false)

    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    v = copy(v_init)
    n_grads = 0
    n_iter = 0
    while n_grads < n_evals
        n_iter += 1
        refreshment!(v,δ/2,refr_rate;unit_sphere=vel_on_sphere)
        x += v * δ/2
        grad_V = Vgrad(x)
        n_grads += jump_part_bps_particles!(grad_V, Wgrad, ub_W, x, v, δ, N)
        x += v * δ/2
        refreshment!(v,δ/2,refr_rate;unit_sphere=vel_on_sphere)
        eval_x = func(x)
        old_est_mean = est_mean
        est_mean += (1/n_iter) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
    end

    est_mean, est_var/(n_iter)

end

# function splitting_bps_particles(
#     Vgrad::Function, # gradient for the V part of the potential
#     Wgrad::Function, # gradient for the W part of the potential
#     func::Function,
#     ub_W::Real,
#     δ::Float64,
#     N::Integer,
#     n_iters::Integer,
#     n_btw::Integer,
#     x_init::Vector{<:Real},
#     v_init::Vector{<:Real},
#     refr_rate::Real = 1.;
#     vel_on_sphere::Bool = false)

#     est_mean = 0.
#     est_var = 0.
#     estims_vec = Vector{Tuple{Float64,Float64}}()
#     x = copy(x_init)
#     v = copy(v_init)
#     n_outerloop = Int(ceil(n_iters / n_btw))
#     n_iter = 0
#     grad_V = Vector{Float64}(undef,N)
#     for i in 1:n_outerloop
#         for j in 1:n_btw        
#             n_iter += 1
#             refreshment!(v,δ/2,refr_rate;unit_sphere=vel_on_sphere)
#             x += v * δ/2
#             grad_V[:] = Vgrad(x)
#             jump_part_bps_particles!(grad_V, Wgrad, ub_W, x, v, δ, N)
#             x += v * δ/2
#             refreshment!(v,δ/2,refr_rate;unit_sphere=vel_on_sphere)
#             eval_x = func(x)
#             old_est_mean = est_mean
#             est_mean += (1/n_iter) * ( eval_x - est_mean )
#             est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
#         end
#         push!(estims_vec,(est_mean,est_var/n_iter))
#     end

#     estims_vec

# end

function splitting_bps_particles(
    Vgrad::Function, # gradient for the V part of the potential
    Wgrad::Function, # gradient for the W part of the potential
    func::Function,
    ub_W::Real,
    δ::Float64,
    N::Integer,
    n_iters::Integer,
    n_btw::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{<:Real},
    refr_rate::Real = 1.  # refreshments here are from the standard Gaussian distribution
    )

    est_mean = 0.
    est_var = 0.
    estims_vec = Vector{Tuple{Float64,Float64}}()
    x = copy(x_init)
    v = copy(v_init)
    n_outerloop = Int(ceil(n_iters / n_btw))
    n_iter = 0
    grad_V = Vector{Float64}(undef,N)
    t_to_refresh = draw_exponential_time(refr_rate)
    for i in 1:n_outerloop
        for j in 1:n_btw        
            n_iter += 1
            t_to_refresh = refreshment_gauss!(v,δ/2,refr_rate,t_to_refresh)
            x += v * δ/2
            grad_V[:] = Vgrad(x)
            jump_part_bps_particles!(grad_V, Wgrad, ub_W, x, v, δ, N)
            x += v * δ/2
            t_to_refresh = refreshment_gauss!(v,δ/2,refr_rate,t_to_refresh)
            eval_x = func(x)
            old_est_mean = est_mean
            est_mean += (1/n_iter) * ( eval_x - est_mean )
            est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        end
        push!(estims_vec,(est_mean,est_var/n_iter))
    end

    estims_vec

end

## Moves 
function jump_part_particles!(grad1::Vector{<:Real}, rate2::Function, ub2::Real, x::Vector{<:Real}, 
    v::Vector{<:Real}, δ::Real, N::Integer)

    grad_count = N
    for i = 1 : N
        t = 0
        rate1 = v[i] * grad1[i]
        time_1 = rate1 > 0 ? (-log(rand()) / (rate1)) : Inf
        # time_1 = draw_exponential_time(rate1) #draw_exponential_time can handle negative rates
        time_2 =  - log(rand()) / (ub2)
        while min(time_1,time_2) <= δ-t
            if time_1 < time_2
                t += time_1
                v[i] *= -1
                time_2 -= time_1
                time_1 = Inf
            else
                t += time_2
                J = rand(1:N)
                true_rate = rate2(x,v,i,J) 
                grad_count += 1
                if true_rate > ub2
                    error("Thinning has failed")
                end
                if rand() < (true_rate / ub2)
                    # flip!(v,i)
                    v[i] *= -1
                    rate1 *= -1
                    time_1 = rate1 > 0 ? (-log(rand()) / (rate1)) : Inf
                    # time_1 = draw_exponential_time(rate1)
                else
                    time_1 -= time_2
                end
                time_2 = - log(rand()) / (ub2)
            end
        end
    end
    grad_count
end




# using Distributions
# function jump_part_particles_seq!(grad1::Vector{<:Real}, rate2::Function, ub2::Real, x::Vector{<:Real}, 
#     v::Vector{<:Real}, δ::Real, N::Integer)

#     sumrates = sum(max.(0,v .* grad1))
#     time_1 =  - log(rand()) / sumrates
#     time_2 =  - log(rand()) / (N * ub2)
#     t = 0.
#     while min(time_1,time_2) <= δ-t
#         if time_1 < time_2
#             t += time_1
#             dist = Categorical(max.(0,v .* grad1) / sumrates)
#             i1_min = rand(dist)
#             sumrates -= v[i1_min] * grad1[i1_min]
#             v[i1_min] *= -1
#             time_2 -= time_1
#             time_1 = draw_exponential_time(sumrates)
#         else
#             t += time_2
#             I,J = rand(1:N,2)
#             true_rate = rate2(x,v,I,J) 
#             if true_rate > ub2
#                 error("Thinning has failed")
#             end
#             if rand() < (true_rate / (ub2))
#                 sumrates -= v[I] * grad1[I]
#                 v[I] *= -1
#                 time_1 = draw_exponential_time(sumrates)
#             else
#                 time_1 -= time_2
#             end
#             time_2 = - log(rand()) / (N * ub2)
#         end
#     end

# end


## previous version -- runs in 680 ns
# function jump_part_particles_seq!(grad1::Vector{<:Real}, rate2::Function, ub2::Real, x::Vector{<:Real}, 
#     v::Vector{<:Real}, δ::Real, N::Integer)

#     # grad_count = N
#     rate1 = v .* grad1
#     timesvec_1 = draw_exponential_time.(rate1)
#     # timesvec_1 = - log.(rand(N)) ./ max.(0,v.*grad1)
#     time_2 = - log(rand()) / (N * ub2)
#     t = 0.
#     (time_1,i1_min) = findmin(timesvec_1)
#     while min(time_1,time_2) <= δ-t
#         if time_1 < time_2
#             t += time_1
#             v[i1_min] *= -1
#             time_2 -= time_1
#             timesvec_1 .-= time_1
#             timesvec_1[i1_min] = Inf
#             (time_1,i1_min) = findmin(timesvec_1)
#         else
#             t += time_2
#             I,J = rand(1:N,2)
#             # J = rand(1:N)
#             true_rate = rate2(x,v,I,J) 
#             # grad_count += 1
#             if true_rate > ub2
#                 error("Thinning has failed")
#             end
#             if rand() < (true_rate / ub2)
#                 v[I] *= -1
#                 rate1[I] *= -1
#                 timesvec_1[I] = rate1[I] > 0 ? (- log(rand()) / (rate1[I])) : Inf
#                 if I == i1_min
#                     (time_1,i1_min) = findmin(timesvec_1)
#                 end
#             else
#                 timesvec_1 .-= time_2
#                 time_1 -= time_2
#             end
#             time_2 = - log(rand()) / (N * ub2)
#         end
#     end

#     # grad_count
# end


# function jump_part_bps_particles!(grad1::Vector{<:Real}, grad2::Function, ub2::Real, x::Vector{<:Real}, 
#     v::Vector{<:Real}, δ::Real, N::Integer)

#     n_grads = 1     # one gradient computation to obtain grad1
#     t = 0.
#     time_1 = draw_exponential_time(max(0,dot(grad1,v)))
#     thin_bound_2 = ub2 * sqrt(N-1) * norm(v)
#     time_2 = draw_exponential_time(thin_bound_2)
#     while t < δ
#         if min(time_1,time_2) <= δ-t
#             if time_1 < time_2
#                 t += time_1
#                 reflect!(grad1,v)
#                 time_2 -= time_1
#                 time_1 = Inf
#             else
#                 t += time_2
#                 # J = rand(1:N)
#                 grad_J = grad2(x,rand(1:N))
#                 actual_rate = dot(grad_J,v) # no need to take max with 0 here
#                 n_grads += 1
#                 if actual_rate > thin_bound_2
#                     error("Failure of the Poisson thinning step. The real reflection rate for W  is $actual_rate, while the upper bound is $thin_bound_2.")
#                 end
#                 if rand() < actual_rate / thin_bound_2 
#                     reflect!(grad_J,v)
#                     time_1 = draw_exponential_time(rate_bps(grad1,v))
#                 else
#                     time_1 -= time_2
#                     # time_1 = draw_exponential_time(rate_bps(grad1(x),v))
#                 end
#                 time_2 = draw_exponential_time(thin_bound_2)
#             end
#         else
#             t = δ+1
#         end
#     end
#     n_grads
# end

function jump_part_bps_particles!(grad1::Vector{<:Real}, grad2::Function, ub2::Real, x::Vector{<:Real}, 
    v::Vector{<:Real}, δ::Real, N::Integer)

    n_grads = 1     # one gradient computation to obtain grad1
    t = 0.
    rate1 = dot(grad1,v)
    time_1 = rate1 > 0 ? (-log(rand()) / (rate1)) : Inf
    thin_bound_2 = ub2 * sqrt(N-1) * norm(v)
    time_2 = - log(rand()) / thin_bound_2
    while t < δ
        if min(time_1,time_2) <= δ-t
            if time_1 < time_2
                t += time_1
                v[:] -= 2 * (rate1 / (dot(grad1,grad1))) * grad1
                # reflect!(grad1,v)
                time_2 -= time_1
                time_1 = Inf
            else
                t += time_2
                # J = rand(1:N)
                grad_J = grad2(x,rand(1:N))
                actual_rate = dot(grad_J,v) 
                n_grads += 1
                if actual_rate > thin_bound_2
                    error("Failure of the Poisson thinning step. The real reflection rate for W  is $actual_rate, while the upper bound is $thin_bound_2.")
                end
                if rand() < actual_rate / thin_bound_2 
                    v[:] -= 2 * (actual_rate / (dot(grad_J,grad_J))) * grad_J
                    # reflect!(grad_J,v)
                    # time_1 = draw_exponential_time(dot(grad1,v))
                    rate1 = dot(grad1,v)
                    time_1 = rate1 > 0 ? (-log(rand()) / (rate1)) : Inf
                else
                    time_1 -= time_2
                end
                time_2 = -log(rand()) / thin_bound_2
            end
        else
            t = δ+1
        end
    end
    n_grads
end

## Functions to run experiments 

function long_run(
    sampler::Function,
    initial_state::Function,
    tolerance::Real,
    iter_tolerance::Integer,
    max_iters::Integer,
    func::Function;
    burn_in::Integer = -1,
    )

    n_iter = 0
    est_mean = 0.0
    est_var = 0.0
    estimates = (0.0,0.0)
    old_estimates = (0.0,0.0)
    z = initial_state()
    change = tolerance + 1

    if burn_in > 1
        println("Starting the burn in phase")
        for k in ProgressBar(1:burn_in)
            z = sampler(z)
        end
        println("Finished the burn in phase")
    end

    for j in ProgressBar(1:Int(ceil(max_iters/iter_tolerance)))

        for n in 1 : iter_tolerance
            z = sampler(z)
            evals = sampler(cond_init)
            eval_x = func(z)
            old_est_mean = est_mean
            est_mean += (1/((j-1)*iter_tolerance + n)) * ( eval_x - est_mean )
            est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        end

        n_iter += iter_tolerance
        estimates = (est_mean, est_var/(n_iter-1))
        change = maximum(abs.((estimates.-old_estimates)./estimates))
        print("Current estimates: $estimates, current relative change: $change \r")
        old_estimates = estimates
        if change < tolerance
            break
        end

    end
    estimates

end


# function run_by_delta(
#     sampler::Function,
#     initial_state::Function,
#     δ::Vector{Float64},
#     n_exp::Int64,
#     truth::Float64
#     )

#     n_delta    = length(δ)
#     errors_samplers = Array{Float64}(undef, n_exp, n_delta)
#     println("")
#     Threads.@threads for k = 1 : n_delta
#         println("Starting with value of step size nr $k out of $n_delta")
#         for j = 1 : n_exp
#             cond_init = initial_state()
#             evals = sampler(δ[k],cond_init)
#             errors_samplers[j,k] = (mean(evals) - truth)^2
#         end
#         println("Finished with value of step size nr $k out of $n_delta")
#     end
#     errors_samplers
# end

function run_by_delta(
    sampler::Function,
    initial_state::Function,
    δ::Vector{Float64},
    n_exp::Int64,
    truth::Tuple{Float64,Float64}
    )

    n_delta    = length(δ)
    # errors_samplers = Array{Tuple{Float64, Float64}}(undef, n_exp, n_delta)
    errors_samplers = (Matrix{Real}(undef, n_exp, n_delta), Matrix{Real}(undef, n_exp, n_delta))
    println("")
    Threads.@threads for j in ProgressBar(1 : n_exp)
        for k in 1:n_delta
            cond_init = initial_state()
            estims = sampler(δ[k],cond_init)
            errors_samplers[1][j,k] = (estims[1] .- truth[1]).^2 / truth[1]^2
            errors_samplers[2][j,k] = (estims[2] .- truth[2]).^2 / truth[2]^2
        end
        # println("Finished with value of step size nr $k out of $n_delta")
    end
    errors_samplers
end

function run_by_iters(
    sampler::Function,
    initial_state::Function,
    δ::Float64,
    n_exp::Int64,
    n_obs::Integer,
    truth::Tuple{Float64,Float64}
    )

    errors_samplers = (Matrix{Real}(undef, n_exp, n_obs), Matrix{Real}(undef, n_exp, n_obs))
    runtimes = zeros(n_exp)
    for j in ProgressBar(1 : n_exp)
            cond_init = initial_state()
            runtimes[j] = @elapsed(estims = sampler(δ,cond_init))
            errors_samplers[1][j,:] = (first.(estims) .- truth[1]).^2 / truth[1]^2
            errors_samplers[2][j,:] = (last.(estims) .- truth[2]).^2 / truth[2]^2
    end
    (errors_samplers,runtimes)
end

function grid_search(
    sampler::Function,
    initial_state::Function,
    vals_iter::AbstractArray,
    n_exp::Int64,
    truth::Tuple{Float64,Float64}
    ) 

    errors = (Array{Real}(undef, n_exp, size(vals_iter)...), Array{Real}(undef, n_exp, size(vals_iter)...))
    Threads.@threads for ind in ProgressBar(CartesianIndices(vals_iter))
        for j = 1 : n_exp
            vals = vals_iter[Tuple(ind)...]
            cond_init = initial_state()
            estims = sampler(vals_iter[ind],cond_init)
            errors[1][j,Tuple(ind)...] = (estims[1] .- truth[1]).^2 / truth[1]^2
            errors[2][j,Tuple(ind)...] = (estims[2] .- truth[2]).^2 / truth[2]^2
        end
    end

    errors

end

function grid_search_runtimes(
    sampler::Function,
    initial_state::Function,
    vals_iter::AbstractArray,
    n_exp::Int64,
    n_obs::Integer,
    truth::Tuple{Float64,Float64}
    ) 
    errors = Array{Tuple{Matrix{Float64}, Matrix{Float64}}}(undef, size(vals_iter)...)
    runtimes = Array{Vector{Float64}}(undef, size(vals_iter)...)
    for ind in ProgressBar(CartesianIndices(vals_iter))
        vals = vals_iter[Tuple(ind)...]
        (errors[Tuple(ind)...],runtimes[Tuple(ind)...]) = run_by_iters(sampler, initial_state, vals, n_exp, n_obs, truth)
    end

    (errors,runtimes)

end

function plot_errors_grid_HMC(errors::Any, vals_iter::AbstractArray)

    deltas = [vals_iter[:, 1, 1][i][1] for i in 1:size(vals_iter,1)]
    Ks = [vals_iter[1, :, 1][i][2] for i in 1:size(vals_iter,2)]
    Ms = [vals_iter[1, 1, :][i][3] for i in 1:size(vals_iter,3)]
    Ks_and_Ms = collect(Iterators.product(Ks,Ms))
    pl_mean = plot()
    pl_var = plot()

    for ind in CartesianIndices(Ks_and_Ms)
        val = Ks_and_Ms[Tuple(ind)...]
        plot!(pl_mean, deltas,mean(errors[1][:,:,Tuple(ind)...]; dims = 1)',xaxis=:log,yaxis=:log, label="(K,M)=$val",  lw = 2)
        plot!(pl_var,  deltas,mean(errors[2][:,:,Tuple(ind)...]; dims = 1)',xaxis=:log,yaxis=:log, label="(K,M)=$val",  lw = 2)
    end
    plot!(pl_mean,legend=:topright, title = "rel-MSE for mean")
    plot!(pl_var,legend=:topright, title = "rel-MSE for variance")
    p_double = plot(
        pl_mean,  pl_var,
        layout = (1, 2),  # 1 row, 2 columns
        size = (600, 400) # Optional: Set figure size
    )
    display(p_double)
    p_double

end

function plot_errors_grid_BPS(errors::Any, vals_iter::AbstractArray)

    deltas = [vals_iter[:, 1][i][1] for i in 1:size(vals_iter,1)]
    refs = [vals_iter[1, :, 1][i][2] for i in 1:size(vals_iter,2)]
    pl_mean = plot()
    pl_var = plot()

    for (ind,val) in enumerate(refs)
        plot!(pl_mean, deltas,mean(errors[1][:,:,Tuple(ind)...]; dims = 1)',xaxis=:log,yaxis=:log, label="Refr rate =$val",  lw = 2)
        plot!(pl_var,  deltas,mean(errors[2][:,:,Tuple(ind)...]; dims = 1)',xaxis=:log,yaxis=:log, label="Refr rate =$val",  lw = 2)
    end
    plot!(pl_mean,legend=:topright, title = "rel-MSE for mean")
    plot!(pl_var,legend=:topright, title = "rel-MSE for variance")
    p_double = plot(
        pl_mean,  pl_var,
        layout = (1, 2),  # 1 row, 2 columns
        size = (600, 400) # Optional: Set figure size
    )
    display(p_double)
    p_double

end

## To save and load data for our experiments

using DataFrames

function add_entries_errors!(df::DataFrame, errs::Any,
    name_sampler::Vector{String}, deltas::Any)

    for j in eachindex(name_sampler)
        for k in 1 : size(errs[1][1],1) # over experiments
            for i in 1 : size(errs[1][1],2) # over step sizes
                push!(
                    df,
                    Dict(
                        :sampler => name_sampler[j],
                        :step_size => deltas[j][i],
                        :error_mean => errs[j][1][k,i],
                        :error_variance => errs[j][2][k,i],
                        ),
                    )
            end
        end
    end

end

function add_entries_errors_runtime!(df::DataFrame, errs::Any,
    name_sampler::Vector{String}, meanruntimes::Vector{Float64})

    for j in eachindex(name_sampler)
        for i in 1 : size(errs[j][1],2) # over runtimes
            push!(
                df,
                Dict(
                    :sampler => name_sampler[j],
                    :runtime => meanruntimes[j],
                    :error_mean => errs[j][1][1,i],
                    :error_variance => errs[j][2][1,i],
                    ),
                )
        end
    end

end

function read_data_give_errors_and_stepsizes(df1::DataFrame)

    names_samplers = unique(df1.sampler)
    deltas_zigzag = unique(df1[df1.sampler .== names_samplers[1],:step_size])
    deltas_bps = unique(df1[df1.sampler .== names_samplers[2],:step_size])
    deltas_HMC_laplace = unique(df1[df1.sampler .== names_samplers[3],:step_size])
    all_deltas = (deltas_zigzag,deltas_bps,deltas_HMC_laplace)
    n_exp = nrow(df1[(df1.sampler .== names_samplers[1]) .& (df1.step_size .==  deltas_zigzag[1]),:])
    
    empty_matrix() = Matrix{Real}(undef, n_exp,length(deltas_zigzag))

    errors = ((empty_matrix(), empty_matrix()),
                (empty_matrix(), empty_matrix()),
                (empty_matrix(), empty_matrix()))

    # errors = Tuple{Tuple{Matrix{Real}, Matrix{Real}}, Tuple{Matrix{Real}, Matrix{Real}}, Tuple{Matrix{Real}, Matrix{Real}}}(undef, )
    for (i,nam) in enumerate(names_samplers)
        for (j,ss) in enumerate(all_deltas[i]) 
            errors[i][1][:,j] = df1[(df1.sampler .== nam) .& (df1.step_size .==  ss), :error_mean]
            errors[i][2][:,j] = df1[(df1.sampler .== nam) .& (df1.step_size .==  ss), :error_variance]
        end
    end
    
    (errors,all_deltas,names_samplers)

end

function read_data_give_errors_and_runtimes(df1::DataFrame)

    names_samplers = unique(df1.sampler)
    runtimes_zigzag = unique(df1[df1.sampler .== names_samplers[1],:runtime])
    runtimes_bps = unique(df1[df1.sampler .== names_samplers[2],:runtime])
    runtimes_HMC_laplace = unique(df1[df1.sampler .== names_samplers[3],:runtime])
    all_runtimes = [runtimes_zigzag,runtimes_bps,runtimes_HMC_laplace]
    length_zigzag = nrow(df1[df1.sampler .== names_samplers[1],:])
    length_bps = nrow(df1[df1.sampler .== names_samplers[2],:])
    length_HMC_laplace = nrow(df1[df1.sampler .== names_samplers[3],:])

    empty_array(r) = Vector{Real}(undef, r)

    # errors = ((empty_array(length_zigzag), empty_array(length_zigzag)),
    #             (empty_array(length_bps), empty_array(length_bps)),
    #             (empty_array(length_HMC_laplace), empty_array(length_HMC_laplace)))
    errors = [[empty_array(length_zigzag), empty_array(length_zigzag)],
                [empty_array(length_bps), empty_array(length_bps)],
                [empty_array(length_HMC_laplace), empty_array(length_HMC_laplace)]]

    for (i,nam) in enumerate(names_samplers)
        errors[i][1] = df1[(df1.sampler .== nam), :error_mean]
        errors[i][2] = df1[(df1.sampler .== nam), :error_variance]
    end
    
    (errors,all_runtimes,names_samplers)

end