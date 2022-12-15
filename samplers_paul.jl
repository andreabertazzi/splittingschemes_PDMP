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
        flip_given_rate!(v, switch_rate_old, δ) #define this function
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
        Z = rand(1)[1]
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

function splitting_zzs_DBD(
                         ∇U::Function,
                         δ::Float64,
                         N::Integer,
                         x_init::Vector{Float64},
                         v_init::Vector{Int64})

    chain = skeleton[]
    push!(chain, skeleton(x_init, v_init, 0,x_init))
    x = copy(x_init)
    v = copy(v_init)
    M=copy(x_init)
    for n = 1 : N
        x = x + v * δ/2
        grad_x = ∇U(x)
        switch_rate_old = max.(0,v.*grad_x)
        flip_given_rate!(v, switch_rate_old, δ) #define this function
        switch_rate_new = max.(0,-v.*grad_x)
        x = x + v * δ/2
        M=M+(x-M)/n;
        push!(chain, skeleton(copy(x), copy(v), n * δ,M))
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
