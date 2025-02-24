function cshift(x,L)
    N=length(x)
    y = zeros(N)

    if L==0
        y=x
    else
        L=Int(-L)
        y[1:N-L]=x[L+1:N]
        y[N-L+1:N]=x[1:L]
    end
    return y
end

function mse(x,y)
    m=sum((x-y).^2)/length(x)
    return m
end

function psnr(x,y)
    m=max(maximum(abs.(x)),maximum(abs.(y)))
    p=20*log10(m)-10*log10(mse(x,y))
    return p
end
