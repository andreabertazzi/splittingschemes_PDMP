include("helper_split.jl")
include("moves_samplers.jl")
include("algorithms.jl")


function jump_part_particles!(rate1::Function, rate2::Function, ub2::Real, x::Vector{<:Real}, 
                                v::Vector{<:Real}, δ::Real, N::Integer)

for i = 1 : N
    t = 0
    time_1 = draw_exponential_time(rate1(x,v,i))
    time_2 = draw_exponential_time(ub2)
    while min(time_1,time_2) <= δ-t
            if time_1 < time_2
                t += time_1
                # println("have switched because of V in particle $i")
                # println("current velocity is ", v[i])
                flip!(v,i)
                # println("new velocity is ", v[i])
                time_2 -= time_1
                time_1 = Inf
            else
                # println("have switched because of W in particle $i")
                t += time_2
                # J = rand(1:(N-1))
                J = rand(1:N)
                true_rate = rate2(x,v,i,J) 
                if true_rate > ub2
                    error("Thinning has failed")
                elseif rand() < (true_rate / ub2)
                    flip!(v,i)
                    time_1 = draw_exponential_time(rate1(x,v,i))
                else
                    time_1 -= time_2
                end
                time_2 = draw_exponential_time(ub2)
            end
    end
end

end


function define_Vrates(Vprime::Function, x::Vector{<:Real}, v::Vector{<:Real}, i::Integer, N::Integer)
    if i == 1
      return pos_part(v[1] * Vprime(x[1]-x[2]))
    elseif i < N
        return pos_part(v[i] * ( Vprime(x[i]-x[i+1]) - Vprime(x[i-1]-x[i]) ))
    elseif i==N
      return pos_part(-v[N] * Vprime(x[N-1]-x[N]))
    else
      error("Index larger than number of particles")
    end
end
  
function define_Wrates(Wprime::Function, x::Vector{<:Real}, v::Vector{<:Real}, i::Integer, j::Integer, N::Integer)
    pos_part( v[i] * Wprime(x[i]-x[j]) ) 
end


function splitting_zzs_particles(
    Vrates::Function,
    Wrates::Function,
    ub_W::Real,
    δ::Float64,
    N::Integer,
    iter::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{Int64})

    chain = Vector{skeleton}(undef,iter+1)
    chain[1] = skeleton(copy(x_init), copy(v_init), 0);
    x = copy(x_init)
    v = copy(v_init)
    for n = 1 : iter
       x = x + v * δ/2
       jump_part_particles!(Vrates, Wrates, ub_W, x, v, δ, N)
       x = x + v * δ/2
       chain[n+1] = skeleton(copy(x), copy(v), n * δ);
    end

    chain

end


function splitting_zzs_particles(
    Vrates::Function,
    Wrates::Function,
    ub_W::Real,
    δ::Float64,
    N::Integer,
    iter::Integer,
    thin_iter::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{Int64})

    if thin_iter == 1
        splitting_zzs_particles(Vrates,Wrates,ub_W,δ,N,iter,x_init,v_init)
    else
        thinned_splitting_zzs_particles(Vrates,Wrates,ub_W,δ,N,iter,thin_iter,x_init,v_init)
    end

end


function thinned_splitting_zzs_particles(
    Vrates::Function,
    Wrates::Function,
    ub_W::Real,
    δ::Float64,
    N::Integer,
    iter::Integer,
    num_thin::Integer,
    x_init::Vector{<:Real},
    v_init::Vector{Int64})

    chain = Vector{skeleton}(undef,num_thin+1)
    chain[1] = skeleton(copy(x_init), copy(v_init), 0);
    x = copy(x_init)
    v = copy(v_init)
    for j = 1 : num_thin
        for n = 1 : iter
            x = x + v * δ/2
            jump_part_particles!(Vrates, Wrates, ub_W, x, v, δ, N)
            x = x + v * δ/2
        end
        chain[j+1] = skeleton(copy(x), copy(v), iter * j * δ);
        print("Simulation progress for ZZS: ", floor(j/num_thin*100), "% \r")
    end
    chain

end

function compute_variance_particles(skele::Vector{skeleton}, barycentre::Vector{<:Real})

    variance = Vector{Float64}(undef, 0)
    N = size(skele[1].position,1)
    for i = 1 : length(skele)
        temp = sum((skele[i].position .- barycentre[i]).^2) / N
        push!(variance, copy(temp))
    end
    variance 
end