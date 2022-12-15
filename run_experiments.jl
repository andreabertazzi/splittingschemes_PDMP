include("helper_split.jl")
include("algorithms.jl")
# include("AdapPDMPs.jl")

function compute_errors(chain::Vector{skeleton}, true_mean::Real, true_rad::Real)
    pos = getPosition(chain, want_array=true)
    μ = mean(pos[1,:])
    err_μ= abs(μ-true_mean)
    rad = mean(sum(pos.^2; dims=1))
    err_rad = abs(rad - true_rad)
    (err_μ,err_rad)
end

function compute_errors(chain::Vector{skeleton}, test_func::Function, truth::Real)
    pos = getPosition(chain, want_array=true)
    est = [test_func(pos[:,j]) for j=1:length(chain)]
    err = abs(mean(est)-truth)
    err
end

function compute_errors(chain::Vector{skeleton}, test_func::Vector{Function}, truth::Vector{Float64})
    pos = getPosition(chain, want_array=true)
    # firstpos = pos[:,1]
    # println("First position is $firstpos")
    n = length(chain)
    n_funcs = length(test_func)
    err = Vector{Float64}(undef, n_funcs)
    for m=1:n_funcs
        vals = [test_func[m].(pos[:,j]) for j=1:n]
        err[m] = abs(mean(vals) - truth[m])
        # cur = err[m]
        # println("Error is $cur")
    end
    err
end

function compute_errors_cts(chain::Vector{skeleton}, test_func::Vector{Function}, truth::Vector{Float64}; step_size::Real = 0.1)
    pos = discretise(chain, step_size, chain[end].time)
    n_funcs = length(test_func)
    err = Vector{Real}(undef, n_funcs)
    for m=1:n_funcs
        vals = [test_func[m].(pos[:,j]) for j=1:n]
        err[m] = abs(mean(vals)-truth[m])
    end
    err
end

function compute_errors_cts(chain::Vector{skeleton}, test_func::Function, truth::Float64; step_size::Real = 0.1)
    pos = discretise(chain, step_size, chain[end].time)
    vals = [test_func(pos[:,j]) for j=1:size(pos,2)]
    err = abs(mean(vals)-truth)
    err
end

function run_by_dims_steps_lengths(
        def_gradient::Function,
        euler_approx::Function,
        splitting_approx::Function,
        initial_v::Function,
        dims::Vector{Int64},
        δ::Vector{Float64},
        N::Vector{Int64},
        n_exp::Int64,
        true_mean::Real,
        true_rad::Vector{Real}
        )

    n_deltas = length(δ)
    n_dims = length(dims)
    err_μ_split = Array{Float64}(undef, n_deltas, n_exp, n_dims)
    error_rad_split = Array{Float64}(undef, n_deltas, n_exp, n_dims)
    err_μ_euler = Array{Float64}(undef, n_deltas, n_exp, n_dims)
    error_rad_euler = Array{Float64}(undef, n_deltas, n_exp, n_dims)

    # If N is integer, then it is a fixed computational budget
    # otherwise it is chosen to ensure the same time horizon for different step sizes.
    if length(N)==1
        nr_steps = Vector{Int64}(N[1]*ones(n_deltas))
    else
        nr_steps = N
    end

    println("")
    for k = 1 : n_dims
        println("Starting with value of dimension nr $k out of $n_dims")
        ∇U = def_gradient(dims[k])
        for i = 1 : n_deltas
            println("Starting with value of delta nr $i out of $n_deltas")
            for j = 1:n_exp
                x_init = rand(dims[k]) - 0.5*ones(dims[k])
                v_init = initial_v(dims[k])
                chain_splitting = splitting_approx(∇U,δ[i],nr_steps[i],x_init,v_init)
                (err_μ_split[i,j,k],error_rad_split[i,j,k]) = compute_errors(chain_splitting, true_mean, true_rad[k])
                chain_euler = euler_approx(∇U,δ[i],nr_steps[i],x_init,v_init)
                (err_μ_euler[i,j,k],error_rad_euler[i,j,k]) = compute_errors(chain_euler, true_mean, true_rad[k])
            end

        end
    end
    (err_μ_split, error_rad_split, err_μ_euler, error_rad_euler)
end


function run_by_lipschitzconstant_and_stepsize(
        def_gradient::Function,
        euler_approx::Function,
        splitting_approx::Function,
        initial_v::Function,
        dim::Int64,
        params_lipschitz::Vector{Float64},
        δ::Vector{Float64},
        N::Vector{Int64},
        n_exp::Int64,
        true_mean::Real,
        true_rad::AbstractVector
        )

    n_deltas = length(δ)
    nr_lipconst = length(params_lipschitz)
    err_μ_split = Array{Float64}(undef, n_deltas, n_exp, nr_lipconst)
    error_rad_split = Array{Float64}(undef, n_deltas, n_exp, nr_lipconst)
    err_μ_euler = Array{Float64}(undef, n_deltas, n_exp, nr_lipconst)
    error_rad_euler = Array{Float64}(undef, n_deltas, n_exp, nr_lipconst)

    # If N is integer, then it is a fixed computational budget
    # otherwise it is chosen to ensure the same time horizon for different step sizes.
    if length(N)==1
        nr_steps = Vector{Int64}(N[1]*ones(n_deltas))
    else
        nr_steps = N
    end

    println("")
    for k = 1 : nr_lipconst
        println("Starting with value of Lipschitz constant nr $k")
        ∇U = def_gradient(dim, params_lipschitz[k])  # def_gradient should take integers in input
        for i = 1 : n_deltas
            println("Starting with value of delta nr $i")
            for j = 1:n_exp
                x_init = rand(dim) - 0.5*ones(dim)
                v_init = initial_v(dim)
                chain_splitting = splitting_approx(∇U,δ[i],nr_steps[i],x_init,v_init)
                (err_μ_split[i,j,k],error_rad_split[i,j,k]) = compute_errors(chain_splitting, true_mean, true_rad[k])
                chain_euler = euler_approx(∇U,δ[i],nr_steps[i],x_init,v_init)
                (err_μ_euler[i,j,k],error_rad_euler[i,j,k]) = compute_errors(chain_euler, true_mean, true_rad[k])
            end

        end
    end
    (err_μ_split, error_rad_split, err_μ_euler, error_rad_euler)
end


function compare_splittings_by_stepssizes(
        ∇U::Function,
        euler_approx::Function,
        splitting_approx_1::Function,
        splitting_approx_2::Function,
        splitting_approx_3::Function,
        initial_state::Function,
        dim::Int64,
        δ::Vector{Float64},
        N::Vector{Int64},
        n_exp::Int64,
        true_mean::Real,
        true_rad::Float64
        )

    n_deltas = length(δ)
    err_μ_split_1 = Array{Float64}(undef, n_deltas, n_exp)
    error_rad_split_1 = Array{Float64}(undef, n_deltas, n_exp)
    err_μ_split_2 = Array{Float64}(undef, n_deltas, n_exp)
    error_rad_split_2 = Array{Float64}(undef, n_deltas, n_exp)
    err_μ_split_3 = Array{Float64}(undef, n_deltas, n_exp)
    error_rad_split_3 = Array{Float64}(undef, n_deltas, n_exp)
    err_μ_euler = Array{Float64}(undef, n_deltas, n_exp)
    error_rad_euler = Array{Float64}(undef, n_deltas, n_exp)

    # If N is integer, then it is a fixed computational budget
    # otherwise it is chosen to ensure the same time horizon for different step sizes.
    if length(N)==1
        nr_steps = Vector{Int64}(N[1]*ones(n_deltas))
    else
        nr_steps = N
    end

    println("")

    for i = 1 : n_deltas
        println("Starting with value of delta nr $i out of $n_deltas")
        for j = 1 : n_exp
            (x_init,v_init) = initial_state(dim)
            chain_splitting_1 = splitting_approx_1(∇U,δ[i],nr_steps[i],x_init,v_init)
            (err_μ_split_1[i,j],error_rad_split_1[i,j]) = compute_errors(chain_splitting_1, true_mean, true_rad)
            chain_splitting_2 = splitting_approx_2(∇U,δ[i],nr_steps[i],x_init,v_init)
            (err_μ_split_2[i,j],error_rad_split_2[i,j]) = compute_errors(chain_splitting_2, true_mean, true_rad)
            chain_splitting_3 = splitting_approx_3(∇U,δ[i],nr_steps[i],x_init,v_init)
            (err_μ_split_3[i,j],error_rad_split_3[i,j]) = compute_errors(chain_splitting_3, true_mean, true_rad)
            chain_euler = euler_approx(∇U,δ[i],nr_steps[i],x_init,v_init)
            (err_μ_euler[i,j],error_rad_euler[i,j]) = compute_errors(chain_euler, true_mean, true_rad)
        end

    end

    (err_μ_split_1, error_rad_split_1, err_μ_split_2, error_rad_split_2,
        err_μ_split_3, error_rad_split_3, err_μ_euler, error_rad_euler)
end


function estimate_inv_measure_gauss(
        ∇U::Function,
        euler_approx::Function,
        splitting_approx_1::Function,
        splitting_approx_2::Function,
        splitting_approx_3::Function,
        initial_state::Function,
        dim::Int64,
        δ::Float64,
        N::Int64,
        n_exp::Int64,
        )
# many runs and keep final point
    samples_split_1 = Array{Float64}(undef, dim, n_exp)
    samples_split_2 = Array{Float64}(undef, dim, n_exp)
    samples_split_3 = Array{Float64}(undef, dim, n_exp)
    samples_euler   = Array{Float64}(undef, dim, n_exp)
    println("")
    for j = 1 : n_exp
        println("Running $j-th experiment")
        (x_init,v_init)   = initial_state(dim)
        chain_splitting_1 = splitting_approx_1(∇U,δ,N,x_init,v_init)
        chain_splitting_2 = splitting_approx_2(∇U,δ,N,x_init,v_init)
        chain_splitting_3 = splitting_approx_3(∇U,δ,N,x_init,v_init)
        # chain_euler       = euler_approx(∇U,δ,N,x_init,v_init)
        samples_split_1[:,j] = chain_splitting_1[end].position
        samples_split_2[:,j] = chain_splitting_2[end].position
        samples_split_3[:,j] = chain_splitting_3[end].position
        # samples_euler[:,j]   = chain_euler[end].position
    end

    (samples_split_1,samples_split_2,samples_split_3,samples_euler)

end


function estimate_error_sampler(
        ∇U::Function,
        Q::Matrix{Float64},
        sampler::Function,
        initial_state::Function,
        T::Float64,
        n_exp::Int64,
        test_func::Function,
        truth::Real,
        )

    err = Vector{Float64}(undef, n_exp)

    for i = 1 : n_exp
        (x_init,v_init) = initial_state(dim)
        chain = sampler(∇U, Q, T, x_init, v_init, refresh_rate)
        err[i] = compute_errors_cts(chain, test_func, truth)
    end

    err

end

# function run_by_refreshmentrates(
#         ∇U::Function,
#         euler_approx::Function,
#         splitting_approxims::Vector{Function},
#         initial_state::Function,
#         dim::Int64,
#         δ::Float64,
#         N::Int64,
#         refresh_rates::AbstractVector,
#         n_exp::Int64,
#         true_mean::Real,
#         true_rad::Real
#         )
#
#     n_splittings = length(splitting_approxims)
#     n_rr = length(refresh_rates)
#     err_μ_splits = Array{Float64}(undef, n_splittings, n_exp, n_rr)
#     error_rad_splits = Array{Float64}(undef, n_splittings, n_exp, n_rr)
#     err_μ_euler = Array{Float64}(undef, n_exp, n_rr)
#     error_rad_euler = Array{Float64}(undef, n_exp, n_rr)
#
#     println("")
#     for k = 1 : n_rr
#         println("Starting with value of refreshment rate nr $k out of $n_rr")
#         global refresh_rate = refresh_rates[k]  # refresh_rate is the name used in the various functions
#
#         for j = 1 : n_exp
#             (x_init,v_init) = initial_state(dim)
#             for i = 1:n_splittings
#                 chain_split = splitting_approxims[i](∇U,δ,N,x_init,v_init)
#                 (err_μ_splits[i,j,k],error_rad_splits[i,j,k]) = compute_errors(chain_split, true_mean, true_rad)
#             end
#             chain_euler = euler_approx(∇U,δ,N,x_init,v_init)
#             (err_μ_euler[j,k],error_rad_euler[j,k]) = compute_errors(chain_euler, true_mean, true_rad)
#         end
#
#     end
#     (err_μ_splits, error_rad_splits, err_μ_euler, error_rad_euler)
# end


function run_by_refreshmentrates(
        ∇U::Function,
        samplers::Vector{Function},
        initial_state::Function,
        dim::Int64,
        δ::Float64,
        N::Int64,
        refresh_rates::AbstractVector,
        n_exp::Int64,
        test_funcs::Vector{Function},
        truth::Vector{Float64}
        )

    n_samplers = length(samplers)
    n_rr       = length(refresh_rates)
    n_testfuncs = length(test_funcs)
    errors_samplers = Array{Float64}(undef, n_samplers, n_exp, n_rr, n_testfuncs)

    println("")
    for k = 1 : n_rr
        println("Starting with value of refreshment rate nr $k out of $n_rr")
        global refresh_rate = refresh_rates[k]  # refresh_rate is the name used in the various functions
        for j = 1 : n_exp
            (x_init,v_init) = initial_state(dim)
            for i = 1:n_samplers
                chain = samplers[i](∇U,δ,N,x_init,v_init)
                errors_samplers[i,j,k,:] = compute_errors(chain, test_funcs, truth)
            end
        end
    end
    errors_samplers
end

function run_by_refreshmentrates(
        ∇U::Function,
        samplers::Vector{Function},
        initial_state::Function,
        dim::Int64,
        δ::Float64,
        N::Int64,
        refresh_rates::AbstractVector,
        n_exp::Int64,
        test_funcs::Function,
        truth::Float64
        )

    n_samplers = length(samplers)
    n_rr       = length(refresh_rates)
    errors_samplers = Array{Float64}(undef, n_samplers, n_exp, n_rr)

    println("")
    for k = 1 : n_rr
        println("Starting with value of refreshment rate nr $k out of $n_rr")
        global refresh_rate = refresh_rates[k]  # refresh_rate is the name used in the various functions
        for j = 1 : n_exp
            (x_init,v_init) = initial_state(dim)
            for i = 1:n_samplers
                chain = samplers[i](∇U,δ,N,x_init,v_init)
                errors_samplers[i,j,k] = compute_errors(chain, test_funcs, truth)
            end
        end
    end
    errors_samplers
end

function run_by_delta(
        ∇U::Function,
        samplers::Vector{Function},
        initial_state::Function,
        dim::Int64,
        δ::Vector{Float64},
        N::Vector{Int64},
        refresh_rates::Real,
        n_exp::Int64,
        test_funcs::Function,
        truth::Float64
        )

    n_samplers = length(samplers)
    n_delta    = length(δ)
    errors_samplers = Array{Float64}(undef, n_samplers, n_exp, n_delta)
    global refresh_rate = refresh_rates  # refresh_rate is the name used in the various functions

    println("")
    for k = 1 : n_delta
        println("Starting with value of delta nr $k out of $n_delta")
        for j = 1 : n_exp
            (x_init,v_init) = initial_state(dim)
            for i = 1:n_samplers
                chain = samplers[i](∇U,δ[k],N[k],x_init,v_init)
                errors_samplers[i,j,k] = compute_errors(chain, test_funcs, truth)
            end
        end
    end
    errors_samplers
end

function estimate_error_byrefresh_cts(
        ∇U::Function,
        Q::Matrix{Float64},
        samplers::Vector{Function},
        initial_state::Function,
        T::Real,
        discr_step::Real,
        refresh_rates::AbstractVector,
        n_exp::Int64,
        test_funcs::Vector{Function},
        truth::Vector{Float64}
        )

    n_samplers = length(samplers)
    n_rr       = length(refresh_rates)
    n_testfuncs = length(test_funcs)
    errors = Array{Float64}(undef, n_samplers, n_exp, n_rr, n_testfuncs)

    println("")
    for k = 1 : n_rr
        println("Starting with value of refreshment rate nr $k out of $n_rr")
        global refresh_rate = refresh_rates[k]  # refresh_rate is the name used in the various functions
        for j = 1 : n_exp
            (x_init,v_init) = initial_state(dim)
            for i = 1:n_samplers
                chain = samplers[i](∇U, Q, T, x_init, v_init, refresh_rates[k])
                positions = discretise(chain, discr_step, T)
                errors[i,j,k,:] = compute_errors_cts(chain, test_funcs, truth; step_size = discr_step)
            end
        end
    end
    errors
end

function chech_metropolis(target::Function, ∇U::Function, δ::Real, n_exp::Integer, N::Integer,
                            test_funcs::Vector{Function},
                            truth::Vector{Float64}
                            )

    n_testfuncs = length(test_funcs)
    errors = Array{Float64}(undef, n_exp, n_testfuncs)
    println("")

    for i =1:n_exp
        x_init = randn(1)
        v_init = [rand((-1,1))]
        chain = splitting_zzs_DBD_metropolised(∇U,target,δ,N,x_init,v_init)
        errors[i,:] = compute_errors(chain, test_funcs, truth)
    end

    errors

end

function run_metropolis(target::Function,
                            sampler::Function,
                            draw_v::Function,
                            ∇U::Function,
                            δ::Real,
                            dim::Integer,
                            n_exp::Integer,
                            N::Integer,
                            test_funcs::Vector{Function},
                            truth::Vector{Float64};
                            want_plot::Bool = false
                            )

    n_testfuncs = length(test_funcs)
    errors = Array{Float64}(undef, n_exp, n_testfuncs)
    println("")

    for i =1:n_exp
        x_init = randn(dim)
        v_init = draw_v(dim)
        # v_init = randn(dim)
        # v_init = v_init/norm(v_init)
        chain = sampler(target,∇U,δ,N,x_init,v_init; want_plot = want_plot)
        errors[i,:] = compute_errors(chain, test_funcs, truth)
    end

    errors

end

function count_rejections_metropolis(
                            sampler::Vector{Function},
                            initial_vel::Vector{Function},
                            target::Function,
                            Σ::Function,
                            Σ_inv::Function,
                            ∇U::Function,
                            δ::Vector{Float64},
                            dim::Vector{Int64},
                            ρ::Vector{Float64},
                            n_exp::Integer,
                            N::Integer;
                            # test_funcs::Vector{Function},
                            # truth::Vector{Float64};
                            want_plot::Bool = false
                            )

    # n_testfuncs = length(test_funcs)
    n_samplers  = length(sampler)
    n_corr      = length(ρ)
    n_dim       = length(dim)
    n_delta     = length(δ)
    n_rej = Array{Float64}(undef, n_samplers, n_exp, n_corr, n_dim, n_delta)
    for k = 1:n_corr
        for l = 1:n_dim
            inv_cov  = Σ_inv(ρ[k],dim[l])
            sqrt_cov = sqrt(Σ(ρ[k],dim[l]))
            target_pdf(x) = target(x,inv_cov)
            grad_U(x) = ∇U(x,inv_cov)
            for j = 1:n_samplers
                for i =1:n_exp
                    for m = 1:n_delta
                        x_init = sqrt_cov*randn(dim[l])
                        v_init = initial_vel[j](dim[l])
                        (chain,n_rej[j,i,k,l,m]) = sampler[j](target_pdf,grad_U,δ[m],N,x_init,v_init; want_rej = true)
                    end
                end
            end
        end
    end

    n_rej

end

function compute_errors_discrete_and_cts_gauss(
                            sampler_metropolis::Vector{Function},
                            sampler_unadjusted::Vector{Function},
                            sampler_continuous::Vector{Function},
                            initial_vel_discr::Vector{Function},
                            initial_vel_cts::Vector{Function},
                            target::Function,
                            Σ::Function,
                            Σ_inv::Function,
                            ∇U::Function,
                            δ::Vector{Float64},
                            dim::Vector{Int64},
                            ρ::Vector{Float64},
                            n_exp::Integer,
                            T::Integer,
                            test_func::Function,
                            truth::Array{Float64,2},
                            refresh_rates::Vector{Float64},
                            discr_step_cts::Float64
                            )

    n_samplers_metro  = length(sampler_metropolis)
    n_samplers_unadj  = length(sampler_unadjusted)
    n_samplers_cts    = length(sampler_continuous)
    n_samplers  = n_samplers_metro + n_samplers_unadj + n_samplers_cts
    n_param     = length(ρ)
    n_dim       = length(dim)
    n_delta     = length(δ)
    err     = Array{Float64}(undef, n_samplers_metro+n_samplers_unadj, n_exp, n_param, n_dim, n_delta)
    err_cts = Array{Float64}(undef, n_samplers_cts, n_exp, n_param, n_dim)
    println("")
    for k = 1:n_param
        println("Starting with value of ρ number $k of $n_param")
        for l = 1:n_dim
            println("Starting with value of dim number $l of $n_dim")
            inv_cov  = Σ_inv(ρ[k],dim[l])
            sqrt_cov = sqrt(Σ(ρ[k],dim[l]))
            target_pdf(x) = target(x,inv_cov)
            grad_U(x) = ∇U(x,inv_cov)
            for i =1:n_exp
                x_init = sqrt_cov*randn(dim[l])
                for m = 1:n_delta
                    # println("Starting with value of delta number $m of $n_delta")
                    N = Integer(round(T/δ[m]))
                    for j = 1:n_samplers_metro
                        v_init = initial_vel_discr[j](dim[l])
                        chain = sampler_metropolis[j](target_pdf,grad_U,δ[m],N,x_init,v_init; want_rej = false)
                        err[j,i,k,l,m] = compute_errors(chain, test_func, truth[k,l])
                    end
                    for j = 1:n_samplers_unadj
                        v_init = initial_vel_discr[j](dim[l])
                        chain  = sampler_unadjusted[j](grad_U,δ[m],N,x_init,v_init)
                        err[j+n_samplers_metro,i,k,l,m] = compute_errors(chain, test_func, truth[k,l])
                    end
                end
                for j = 1:n_samplers_cts
                    v_init = initial_vel_cts[j](dim[l])
                    chain = sampler_continuous[j](grad_U, inv_cov, T, x_init, v_init, refresh_rates[j])
                    err_cts[j,i,k,l] = compute_errors_cts(chain, test_func, truth[k,l]; step_size = discr_step_cts)
                end
            end
        end
    end

    (err,err_cts)

end
