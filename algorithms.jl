include("helper_split.jl")
include("moves_samplers.jl")

function splitting_ABA(part_A!::Function,
                        part_B!::Function,
                        ∇U::Function,
                        δ::Float64,
                        N::Integer,
                        x_init::Vector{Float64},
                        v_init::AbstractVector)

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = x_init
    v = v_init
    for n = 1:N
        part_A!(∇U,x,v,δ/2)
        part_B!(∇U,x,v,δ)
        part_A!(∇U,x,v,δ/2)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    chain
end

function splitting_ABA_fun(part_A!::Function,
    part_B!::Function,
    ∇U::Function,
    δ::Float64,
    N::Integer,
    x_init::Vector{Float64},
    v_init::AbstractVector,
    stat_fun::Function)

    estimate = 0
    x = x_init
    v = v_init
    for n = 1:N
        part_A!(∇U,x,v,δ/2)
        part_B!(∇U,x,v,δ)
        part_A!(∇U,x,v,δ/2)
        estimate += stat_fun(x)/N
    end
    estimate
end

function zzs_metropolised(target::Function,
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::Vector{Int64};
                         want_rej::Bool = false,
                         want_plot::Bool = false
                         )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ) 
        switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        num = target(x) * exp(-δ*sum(switch_rate_new))
        den = target(x_old) * exp(-δ*sum(switch_rate_old))
        Z = rand(1)[1]
        if Z > num/den
            # println("Rejection!")
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej+=1
        end
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    # println("")
    # println("Number of rejections: $n_rej")
    if want_plot
        iter = Integer(round(200/δ))
        # times = [chain[i].time for i = 1:iter]
        # positions = [chain[i].position[1] for i = 1:iter]
        # display(plot(times,positions))
        positions1 = [chain[i].position[1] for i = 1:iter]
        positions2 = [chain[i].position[2] for i = 1:iter]
        display(plot(positions1,positions2))
    end

    # chain
    if want_rej
        (chain,n_rej)
    else
        chain
    end

end

function zzs_metropolised_energy(U::Function,
                                ∇U::Function,
                                δ::Float64,
                                N::Integer,
                                x_init::Vector{Float64},
                                v_init::Vector{Int64};
                                want_rej::Bool = false,
                                )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ) 
        switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        log_num = -U(x) - δ*sum(switch_rate_new)
        log_den = -U(x_old) - δ*sum(switch_rate_old)
        Z = rand(1)[1]
        if Z > exp(log_num - log_den)
            # println("Rejection!")
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej+=1
        end
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end

    if want_rej
        (chain,n_rej)
    else
        chain
    end

end

function zzs_metropolised_energy(U::Function,
                                grad_U::Function,
                                func::Function,
                                δ::Float64,
                                n_iter::Integer,
                                x_init::Vector{<:Real},
                                v_init::Vector{<:Real};
                                want_rej::Bool = false,
                                )

    # est = Vector{Real}(undef, n_iter+1)
    # est[1] = func(x_init)
    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for j = 1 : n_iter
        x_old = copy(x)
        v_old = copy(v)
        x += v * δ/2
        grad_x = grad_U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ)
        switch_rate_new = max.(0,-v.*grad_x)
        x += v * δ/2
        if rand() > exp( U(x_old) - U(x) +  δ * sum(switch_rate_old .- switch_rate_new))
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej += 1
        end
        # est[j+1] = func(x)
        eval_x = func(x)
        old_est_mean = est_mean
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
    end
    if want_rej
        (est_mean, est_var / n_iter, n_rej)
    else
        est_mean, est_var/n_iter
    end

end

function zzs_metropolised_energy_RS(U::Function,
    grad_U::Function,
    func::Function,
    δ::Float64,
    n_iter::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{<:Real};
    want_rej::Bool = false,
    )

    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for j = 1 : n_iter
        step_size = draw_exponential_time(1/δ)
        x_old = copy(x)
        v_old = copy(v)
        x += v * step_size/2
        grad_x = grad_U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, step_size)
        switch_rate_new = max.(0,-v.*grad_x)
        x += v * step_size/2
        if rand() > exp( U(x_old) - U(x) +  step_size * sum(switch_rate_old .- switch_rate_new))
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej += 1
        end
        # est[j+1] = func(x)
        eval_x = func(x)
        old_est_mean = est_mean
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
    end
    if want_rej
        (est_mean, est_var / n_iter, n_rej)
    else
        est_mean, est_var/n_iter
    end

end


function zzs_metropolis_randstepsize(target::Function,
    ∇U::Function,
    avgstepsize::Float64,
    N::Integer,
    x_init::Vector{Float64},
    v_init::Vector{Int64}
    )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        v_old = copy(v)
        δ = draw_exponential_time(1/avgstepsize)
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ) 
        switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        num = target(x) * exp(-δ*sum(switch_rate_new))
        den = target(x_old) * exp(-δ*sum(switch_rate_old))
        Z = rand(1)[1]
        if Z > num/den
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej+=1
        end
        push!(chain, skeleton(copy(x), copy(v), n * avgstepsize))
    end

    chain

end


function bps_metropolised(
                         target::Function,
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::AbstractVector;
                         unit_sphere::Bool = false,
                         want_rej::Bool = false,
                         want_plot::Bool = false
                         )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        scal_prod = dot(v,grad_x)
        switch_rate_old = max(0,scal_prod)
        reflect_given_rate!(v, switch_rate_old, grad_x, δ)
        switch_rate_new = max(0,-dot(v,grad_x))
        x = x + v * δ/2
        num = target(x) * exp(-δ*switch_rate_new)
        den = target(x_old) * exp(-δ*switch_rate_old)
        Z = rand()
        if Z > num/den
            # println("Rejection!")
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej+=1
        end
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    if want_plot
        # iter = Integer(round(200/δ))
        iter = 1000
        times = [chain[i].time for i = 1:iter]
        positions = [chain[i].position[2] for i = 1:iter]
        display(plot(times,positions))
        positions1 = [chain[i].position[1] for i = 1:iter]
        positions2 = [chain[i].position[2] for i = 1:iter]
        display(plot(positions1,positions2, label = "Metropolised BPS"))
    end

    if want_rej
        (chain,n_rej)
    else
        chain
    end

end

function bps_metropolised_energy(
    U::Function,
    ∇U::Function,
    δ::Float64,
    N::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{<:Real},
    refr_rate::Real = 1.;
    v_sphere::Bool = false,
    want_rej::Bool = false,
    want_plot::Bool = false
    )

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for n = 1 : N
        x_old = copy(x)
        refreshment!(v,δ/2,refr_rate;unit_sphere=v_sphere)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        scal_prod = dot(v,grad_x)
        switch_rate_old = max(0,scal_prod)
        reflect_given_rate!(v, switch_rate_old, grad_x, δ)
        switch_rate_new = max(0,-dot(v,grad_x))
        x = x + v * δ/2
        log_num = - U(x) - δ * switch_rate_new
        log_den = - U(x_old) - δ * switch_rate_old
        Z = rand()
        if Z > exp(log_num - log_den)
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej +=1
        end
        refreshment!(v,δ/2,refr_rate;unit_sphere=v_sphere)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end

    if want_rej
        (chain,n_rej)
    else
        chain
    end

end

function bps_metropolised_energy_1step(
    U::Function,
    ∇U::Function,
    δ::Float64,
    x_init::Vector{<:Real},
    v_init::Vector{<:Real},
    refr_rate::Real = 1.;
    v_sphere::Bool = false
    )

    chain = skeleton[]
    x = copy(x_init)
    v = copy(v_init)

    x_old = copy(x)
    refreshment!(v,δ/2,refr_rate;unit_sphere=v_sphere)
    v_old = copy(v)
    x = x + v * δ/2
    grad_x = ∇U(x)
    scal_prod = dot(v,grad_x)
    switch_rate_old = max(0,scal_prod)
    reflect_given_rate!(v, switch_rate_old, grad_x, δ)
    switch_rate_new = max(0,-dot(v,grad_x))
    x = x + v * δ/2
    log_num = - U(x) - δ * switch_rate_new
    log_den = - U(x_old) - δ * switch_rate_old
    Z = rand()
    if Z > exp(log_num - log_den)
        (x,v) = (copy(x_old),-copy(v_old))
    end
    refreshment!(v,δ/2,refr_rate;unit_sphere=v_sphere)
    
    return (x,v)

end

function bps_metropolised_energy(
                                U::Function,
                                ∇U::Function,
                                func::Function,
                                δ::Float64,
                                n_iter::Integer,
                                x_init::Vector{<:Real},
                                v_init::Vector{<:Real},
                                refr_rate::Real = 1.;
                                v_sphere::Bool = false,
                                want_rej::Bool = false,
                                )

    # evals = Vector{Real}(undef, n_iter+1)
    # evals[1] = func(x_init)
    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    v = copy(v_init)
    n_rej = 0
    for j = 1 : n_iter
        refreshment!(v,δ/2,refr_rate;unit_sphere=v_sphere)
        x_old = copy(x)
        v_old = copy(v)
        x += v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max(0,dot(v,grad_x))
        reflect_given_rate!(v, switch_rate_old, grad_x, δ)
        switch_rate_new = max(0,-dot(v,grad_x))
        x += v * δ/2
        if rand() > exp( U(x_old) - U(x) +  δ * sum(switch_rate_old .- switch_rate_new))
            (x,v) = (copy(x_old),-copy(v_old))
            n_rej +=1
        end
        refreshment!(v,δ/2,refr_rate;unit_sphere=v_sphere)
        eval_x = func(x)
        old_est_mean = est_mean
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        # evals[j+1] = func(x)
        # print("Simulation progress for BPS: ", floor(j/n_iter*100), "% \r")
    end

    if want_rej
        (est_mean, est_var / n_iter, n_rej)
    else
        est_mean, est_var/n_iter
    end

end

function splitting_zzs_DBD(
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::Vector{Int64})

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ)
        # switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end

    chain

end


function splitting_bps_RDBDR_gauss(
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::AbstractVector;
                         unit_sphere::Bool = false,
                         want_plot::Bool = false)

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        x_old = copy(x)
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        v_old = copy(v)
        x = x + v * δ/2
        grad_x = ∇U(x)
        scal_prod = dot(v,grad_x)
        switch_rate_old = max(0,scal_prod)
        reflect_given_rate!(v, switch_rate_old, grad_x, δ)
        switch_rate_new = max(0,-dot(v,grad_x))
        x = x + v * δ/2
        (x,v) = refreshment_part_bps(x,v,δ/2; unit_sphere = unit_sphere)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    if want_plot
        # iter = Integer(round(200/δ))
        iter = N
        # times = [chain[i].time for i = 1:iter]
        # positions = [chain[i].position[1] for i = 1:iter]
        # display(plot(times,positions))
        positions1 = [chain[i].position[1] for i = 1:iter]
        positions2 = [chain[i].position[2] for i = 1:iter]
        display(plot(positions1,positions2))
    end

    chain

end


function splitting_ABCBA(part_A!::Function,
                         part_B!::Function,
                         part_C!::Function,
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::AbstractVector)

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        part_A!(∇U,x,v,δ/2)
        part_B!(∇U,x,v,δ/2)
        part_C!(∇U,x,v,δ)
        part_B!(∇U,x,v,δ/2)
        part_A!(∇U,x,v,δ/2)
        push!(chain, skeleton(copy(x), copy(v), n * δ))
    end
    chain

end

function splitting_ABCBA_fun(part_A!::Function,
                            part_B!::Function,
                            part_C!::Function,
                            ∇U::Function,
                            δ::Float64,
                            N::Integer,
                            x_init::Vector{Float64},
                            v_init::AbstractVector,
                            stat_fun::Function)

    estimate = 0
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        part_A!(∇U,x,v,δ/2)
        part_B!(∇U,x,v,δ/2)
        part_C!(∇U,x,v,δ)
        part_B!(∇U,x,v,δ/2)
        part_A!(∇U,x,v,δ/2)
        estimate += stat_fun(x) / N
    end
    
    estimate

end

# function splitting_zzs_DBD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::AbstractVector)
#     splitting_ABA(flow_zzs!,jump_part_zzs!,∇U,δ,N,x_init,v_init)
# end

# Splittings of BPS

function splitting_bps_RDBDR(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(refreshment_part_bps!,flow_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_RBDBR(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(refreshment_part_bps!,reflection_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init)
end

# function splitting_bps_RDBDR_gauss(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
#     splitting_ABCBA(refreshment_part_bps_gauss!,flow_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init)
# end

function splitting_bps_DBRBD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(flow_bps!,reflection_part_bps!,refreshment_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DBRBD_gauss(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(flow_bps!,reflection_part_bps!,refreshment_part_bps_gauss!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DRBRD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(flow_bps!,refreshment_part_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BRDRB(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(reflection_part_bps!,refreshment_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BDRDB(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(reflection_part_bps!,flow_bps!,refreshment_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BDRDB_gauss(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABCBA(reflection_part_bps!,flow_bps!,refreshment_part_bps_gauss!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DBD(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(flow_bps!,jump_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_BDB(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(jump_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_DR_B_DR(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(flow_and_refreshment_bps!,jump_part_bps!,∇U,δ,N,x_init,v_init)
end

function splitting_bps_B_DR_B(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64})
    splitting_ABA(jump_part_bps!,flow_and_refreshment_bps!,∇U,δ,N,x_init,v_init)
end


## Splitting schemes that estimate an expected statistic along the way, without storing the chain
function splitting_bps_RDBDR_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABCBA_fun(refreshment_part_bps!,flow_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end

function splitting_bps_RBDBR_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABCBA_fun(refreshment_part_bps!,reflection_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init,stat_fun)
end


function splitting_bps_DRBRD_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABCBA_fun(flow_bps!,refreshment_part_bps!,reflection_part_bps!,∇U,δ,N,x_init,v_init,stat_fun)
end

function splitting_bps_DBRBD_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABCBA_fun(flow_bps!,reflection_part_bps!,refreshment_part_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end

function splitting_bps_BDRDB_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABCBA_fun(reflection_part_bps!,flow_bps!,refreshment_part_bps!,∇U,δ,N,x_init,v_init,stat_fun)
end

function splitting_bps_BRDRB_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABCBA_fun(reflection_part_bps!,refreshment_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end

function splitting_bps_BDRDB_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABCBA_fun(reflection_part_bps!,flow_bps!,refreshment_part_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end

function splitting_bps_DBD_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABA_fun(flow_bps!,jump_part_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end

function splitting_bps_BDB_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABA_fun(jump_part_bps!,flow_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end

function splitting_bps_DR_B_DR_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABA_fun(flow_and_refreshment_bps!,jump_part_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end

function splitting_bps_B_DR_B_fun(∇U::Function,δ::Float64,N::Int64,x_init::Vector{Float64},v_init::Vector{Float64}, stat_fun::Function)
    splitting_ABA_fun(jump_part_bps!,flow_and_refreshment_bps!,∇U,δ,N,x_init,v_init, stat_fun)
end



## others

function euler_pdmp_FD(flow::Function,
                    event_rates::Function,
                    jump::Function,
                    ∇U::Function,
                    δ::Float64,
                    N::Integer,
                    x_init::Vector{Float64},
                    v_init::AbstractVector)
    # Fully Discrete Euler approximation
    # Initial conditions required!
    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for i = 1 : N
        grad = ∇U(x)   # compute this as can re-use it. o.w. more than one grad computation per iteration :(
        rates = event_rates(grad,v) # can be scalar or vector
        t_event, i_event = event_times(rates)
        (x,v) = flow(x,v,δ)
        if t_event <= δ
            (x,v) = jump(grad,x,v,i_event)
        end
        push!(chain, skeleton(copy(x), copy(v), n*δ))
    end
    chain
end

function euler_pdmp_PD(flow::Function,
                    event_rates::Function,
                    jump::Function,
                    ∇U::Function,
                    δ::Float64,
                    N::Integer,
                    x_init::Vector{Float64},
                    v_init::AbstractVector)
    # Partially Discrete Euler approximation
    # Initial conditions required!
    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0))
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : N
        grad = ∇U(x)
        rates = event_rates(grad,v) # can be scalar or vector
        t_event, i_event = event_times(rates)
        if t_event > δ
            (x,v) = flow(x,v,δ)
        else
            (x,v) = flow(x,v,t_event)
            (x,v) = jump(∇U,x,v,i_event)
            (x,v) = flow(x,v,δ-t_event)
        end
        push!(chain, skeleton(copy(x), copy(v), n*δ))
    end
    chain
end

function euler_zzs_FD(∇U::Function,δ::Float64,N::Integer,x_init::Vector{Float64},v_init::Vector{Real})
    euler_pdmp_FD(flow_zzs,event_rates_zzs,jump_zzs,∇U,δ,N,x_init,v_init)
end

function euler_zzs_PD(∇U::Function,δ::Float64,N::Integer,x_init::Vector{Float64},v_init::Vector{Int64})
    euler_pdmp_PD(flow_zzs,event_rates_zzs,jump_zzs,∇U,δ,N,x_init,v_init)
end

function euler_bps_PD(∇U::Function,δ::Float64,N::Integer,x_init::Vector{Float64},v_init::Vector{Float64})
    euler_pdmp_PD(flow_bps,event_rates_bps,jump_bps,∇U,δ,N,x_init,v_init)
end


tolerance = 1e-7

function BPS(∇E::Function, Q::Matrix{Float64}, T::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Float64} = Vector{Float64}(undef,0), refresh_rate::Float64 = 1.0)

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = randn(dim)
    end

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;

    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))

    rejected_switches = 0;
    accepted_switches = 0;
    gradient = ∇E(x);
    a = transpose(v) * gradient;
    b = transpose(v) * Q * v;

    Δt_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)
        gradient = ∇E(x)

        if (!finished && Δt_switch_proposed < Δt_refresh)
            switch_rate = transpose(v) * gradient
            proposedSwitchIntensity = a
            if proposedSwitchIntensity < switch_rate - tolerance
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v = reflect(gradient,v)
                a = -switch_rate
                b = transpose(v) * Q * v
                updateSkeleton = true
                accepted_switches += 1
            else
                a = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update time to refresh
            Δt_refresh = Δt_refresh - Δt_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            a = transpose(v) * gradient
            b = transpose(v) * Q * v

            # update time to refresh
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            # push!(x_skeleton, x)
            # push!(v_skeleton, v)
            # push!(t_skeleton, t)
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end
        Δt_switch_proposed = switchingtime(a,b)
    end
    # println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    # println("number of proposed switches: ", accepted_switches + rejected_switches)
    # return (t_skeleton, x_skeleton, v_skeleton)
    return skel_chain
end



function ZigZag(∇E::Function, Q::Matrix{Float64}, T::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
    # ∂E(i,x) is the i-th partial derivative of the potential E, evaluated in x
    # Q is a symmetric matrix with nonnegative entries such that |(∇^2 E(x))_{ij}| <= Q_{ij} for all x, i, j
    # T is time horizon
    ∂E(i,x) = ∇E(x)[i]
    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = rand((-1,1), dim)
    end

    b = [norm(Q[:,i]) for i=1:dim];
    b = sqrt(dim)*b;

    t = 0.0;
    x = copy(x_init); v = copy(v_init);
    updateSkeleton = false;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(copy(x),copy(v),t))

    rejected_switches = 0;
    accepted_switches = 0;
    initial_gradient = [∂E(i,x) for i in 1:dim];
    a = v .* initial_gradient

    Δt_proposed_switches = switchingtime.(a,b)
    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    while (!finished)
        i = argmin(Δt_proposed_switches) # O(d)
        Δt_switch_proposed = Δt_proposed_switches[i]
        Δt = min(Δt_switch_proposed,Δt_excess);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_excess)
            switch_rate = v[i] * ∂E(i,x)
            proposedSwitchIntensity = a[i]
            if proposedSwitchIntensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v[i] = -v[i]
                a[i] = -switch_rate
                updateSkeleton = true
                accepted_switches += 1
            else
                a[i] = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_excess = Δt_excess - Δt_switch_proposed
            Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed
            Δt_proposed_switches[i] = switchingtime(a[i],b[i])
        elseif !finished
            # so we switch due to excess switching rate
            updateSkeleton = true
            i = rand(1:dim)
            v[i] = -v[i]
            a[i] = v[i] * ∂E(i,x)

            # update upcoming event times
            Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
            Δt_excess = -log(rand())/(dim*excess_rate);
        end

        if updateSkeleton
            push!(skel_chain,skeleton(copy(x),copy(v),t))
            updateSkeleton = false
        end

    end

    return (skel_chain,accepted_switches,rejected_switches)

end


function ULA(gradU::Function,
                δ::Float64,
                N::Integer,
                x_init::Vector{<:Real},
                )

    chain = Vector{Vector{<:Real}}(undef,N+1)
    chain[1,:] = copy(x_init)
    dim = length(x_init)

    for n = 1:N
        chain[n+1,:] = chain[n,:] - δ * gradU(chain[n,:]) + sqrt(2*δ) * randn(dim)
    end

    chain

end


function MALTA(
    U::Function,
    gradU::Function,
    func::Function,
    δ::Float64,
    D::Real,
    n_iter::Integer,
    x_init::Vector{<:Real},
    )

    # est = Vector{Real}(undef, n_iter+1)
    # est[1] = func(x_init)
    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    L = size(x)
    # D = sqrt(length(x))
    for j = 1 : n_iter
        grad_x = gradU(x)
        proposal = x - δ * D * grad_x / max(D,norm(grad_x)) + sqrt(2*δ) * randn(L)
        grad_prop = gradU(proposal)
        log_num = -U(proposal) - norm(x - proposal + δ * D * grad_prop / max(D,norm(grad_prop)) )^2 / (4 * δ)
        log_den = -U(x)        - norm(proposal - x + δ * D * grad_x / max(D,norm(grad_x)) )^2 / (4 * δ) 
        if rand() <= exp(log_num - log_den)
            x = copy(proposal)
        end
        old_est_mean = est_mean
        eval_x = func(x)
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        # est[j+1] = func(x)
        # print("Simulation progress for MALTA: ", floor(j/num_thin*100), "% \r")
    end
    est_mean, est_var / n_iter

end

function SMALTA_func(
                    grad::Function,
                    func::Function,
                    δ::Float64,
                    N::Integer,
                    D::Real,
                    iter::Integer,
                    x_init::Vector{<:Real},
                    )

    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    # D = sqrt(length(x))
    for j = 1 : iter
        J = rand((1:N))
        grad_x = grad(x,J)
        x += - δ * D * grad_x / max(D,norm(grad_x)) + sqrt(2*δ) * randn(N)
        eval_x = func(x)
        old_est_mean = est_mean
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
    end
    est_mean, est_var / iter

end

function HMC_Laplace(U::Function,
                    grad_U::Function,
                    func::Function,
                    δ::Float64,
                    K::Integer,
                    n_iter::Integer,
                    x_init::Vector{<:Real},
                    v_init::Vector{<:Real}
                    )

    x = copy(x_init)
    v = copy(v_init)
    est_mean = 0.
    est_var = 0.
    dim = length(x)
    old_logtarget = - U(x)
    old_log_hamiltonian = old_logtarget - sum(abs.(v))
    grad_x = grad_U(x)
    old_grad_x = copy(grad_x)
    for j = 1 : n_iter
        x_old = copy(x)
        v_old = copy(v)
        for k = 1 : K
            v -= 0.5 * δ * grad_U(x) 
            x += δ * sign.(v)
            grad_x = grad_U(x)
            v -= 0.5 * δ * grad_U(x) 
        end
        new_logtarget = -U(x)
        new_loghamiltonian = new_logtarget - sum(abs.(v))
        if rand() > (new_loghamiltonian / old_log_hamiltonian)
            x = copy(x_old)
            grad_x = copy(old_grad_x)
        else
            old_logtarget = new_logtarget
            old_grad_x = copy(grad_x)
        end
        v = draw_laplace(dim)
        old_log_hamiltonian = old_logtarget - sum(abs.(v))
        old_est_mean = est_mean
        eval_x = func(x)
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        # est_var = (1/j) * (func(x) - est_mean)^2 + (1 - 1/j) * est_var
    end
    
    est_mean, est_var / ( n_iter - 1 )

end

function HMC_Laplace_RS(U::Function,
    grad_U::Function,
    func::Function,
    δ::Float64,
    K::Integer,
    n_iter::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{<:Real}
    )

    x = copy(x_init)
    v = copy(v_init)
    est_mean = 0.
    est_var = 0.
    dim = length(x)
    old_logtarget = - U(x)
    old_log_hamiltonian = old_logtarget - sum(abs.(v))
    grad_x = grad_U(x)
    old_grad_x = copy(grad_x)
    for j = 1 : n_iter
        step_size = draw_exponential_time(1/δ)
        x_old = copy(x)
        v_old = copy(v)
        for k = 1 : K
            v -= 0.5 * step_size * grad_x 
            x += step_size * sign.(v)
            grad_x = grad_U(x)
            v -= 0.5 * step_size * grad_x
        end
        new_logtarget = -U(x)
        new_loghamiltonian = new_logtarget - sum(abs.(v))
        if rand() > (new_loghamiltonian / old_log_hamiltonian)
            x = copy(x_old)
            grad_x = copy(old_grad_x)
        else
            old_logtarget = new_logtarget
            old_grad_x = copy(grad_x)
        end
        v = draw_laplace(dim)
        old_log_hamiltonian = old_logtarget - sum(abs.(v))
        old_est_mean = est_mean
        eval_x = func(x)
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        # est_var = (1/j) * (func(x) - est_mean)^2 + (1 - 1/j) * est_var
    end

    est_mean, est_var / ( n_iter - 1 )

end


function SHMC_Laplace(
                        grad1::Function,
                        grad2::Function,
                        func::Function,
                        N::Integer,
                        δ::Float64,
                        K::Integer,
                        M::Integer,
                        n_iter::Integer,
                        x_init::Vector{<:Real},
                        v_init::Vector{<:Real}
                        )

    est_mean = 0.
    est_var = 0.
    x = copy(x_init)
    v = copy(v_init)
    grad1_vec = grad1(x) 
    for j = 1 : n_iter
        for k = 1 : M
            J = rand((1:N))
            v -= 0.5 * (K * δ) * grad2(x,J) 
            for l in 1 : K
                v -= 0.5 * δ * grad1_vec
                x += δ * sign.(v)
                grad1_vec = grad1(x) 
                v -= 0.5 * δ * grad1_vec
            end
            J = rand((1:N))
            v -= 0.5 * (K * δ) * grad2(x,J) 
        end
        v = draw_laplace(N)
        eval_x = func(x)
        old_est_mean = est_mean
        est_mean += (1/j) * ( eval_x - est_mean )
        est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
    end

    est_mean, est_var/n_iter

end

function SHMC_Laplace(
                        grad1::Function,
                        grad2::Function,
                        func::Function,
                        N::Integer,
                        δ::Float64,
                        K::Integer,
                        M::Integer,
                        n_iters::Integer,
                        n_btw::Integer,
                        x_init::Vector{<:Real},
                        v_init::Vector{<:Real}
                        )

    est_mean = 0.
    est_var = 0.
    estims_vec = Vector{Tuple{Float64,Float64}}()
    x = copy(x_init)
    v = copy(v_init)
    n_outerloop = Int(ceil(n_iters / n_btw))
    n_iter = 0
    grad1_vec = grad1(x) 
    for i in 1:n_outerloop
        for j in 1:n_btw     
            n_iter += 1
            for k = 1 : M
                J = rand((1:N))
                v -= 0.5 * (K * δ) * grad2(x,J) 
                for l in 1 : K
                    v -= 0.5 * δ * grad1_vec
                    x += δ * sign.(v)
                    grad1_vec = grad1(x) 
                    v -= 0.5 * δ * grad1_vec
                end
                J = rand((1:N))
                v -= 0.5 * (K * δ) * grad2(x,J) 
            end
            v = draw_laplace(N)
            eval_x = func(x)
            old_est_mean = est_mean
            est_mean += (1/n_iter) * ( eval_x - est_mean )
            est_var  +=  ( eval_x - old_est_mean ) * ( eval_x - est_mean )
        end
        push!(estims_vec,(est_mean,est_var/n_iter))
    end

    estims_vec

end
