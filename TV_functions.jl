function wraparound(x, m)
    # Extend x so as to wrap around on both axes, sufficient to allow a
    # "valid" convolution with m to return the cyclical convolution.
    # We assume mask origin near centre of mask for compatibility with
    # "same" option.
    (mx, nx) = size(x);
    (mm, nm) = size(m);
    if mm > mx | nm > nx
        error("Mask does not fit inside array")
    end
    
    mo = floor(Int,(1+mm)/2); no = floor(Int,(1+nm)/2);  # reflected mask origin
    ml = mo-1;            nl = no-1;             # mask left/above origin
    mr = mm-mo;           nr = nm-no;            # mask right/below origin
    me = mx-ml+1;         ne = nx-nl+1;          # reflected margin in input
    mt = mx+ml;           nt = nx+nl;            # top of image in output
    my = mx+mm-1;         ny = nx+nm-1;          # output size
    
    y = zeros(my, ny);
    y[mo:mt, no:nt] = x;      # central region
    if ml > 0
        y[1:ml, no:nt] = x[me:mx, :];                   # top side
        if nl > 0
            y[1:ml, 1:nl] = x[me:mx, ne:nx];            # top left corner
        end
        if nr > 0
            y[1:ml, nt+1:ny] = x[me:mx, 1:nr];          # top right corner
        end
    end
    if mr > 0
        y[mt+1:my, no:nt] = x[1:mr, :];                 # bottom side
        if nl > 0
            y[mt+1:my, 1:nl] = x[1:mr, ne:nx];          # bottom left corner
        end
        if nr > 0
            y[mt+1:my, nt+1:ny] = x[1:mr, 1:nr];        # bottom right corner
        end
    end
    if nl > 0
        y[mo:mt, 1:nl] = x[:, ne:nx];                   # left side
    end
    if nr > 0
        y[mo:mt, nt+1:ny] = x[:, 1:nr];                 # right side
    end
    return y
end


function conv2c(x,h)
    (mx, nx) = size(x);
    (mm, nm) = size(h);   
    mo = floor(Int,(1+mm)/2); no = floor(Int,(1+nm)/2);  # reflected mask origin
    ml = mo-1;            nl = no-1;             # mask left/above origin
    mr = mm-mo;           nr = nm-no;            # mask right/below origin
    me = mx-ml+1;         ne = nx-nl+1;          # reflected margin in input
    mt = mx+ml;           nt = nx+nl;            # top of image in output
    my = mx+mm-1;         ny = nx+nm-1;          # output size
    # Circular 2D convolution
    x=wraparound(x,h);
    #x=padarray(x, Pad(:circular,size(h)))
    y=conv(x,h);
    y=y[mm:mm+mx-1,nm:nm+nx-1]
    #y=imfilter(x,reflect(h),Fill(0,h))
    #y=imfilter(x,reflect(h),"circular")
    return y
end

function diffh(x)
    h=[0 1 -1];
    sol=conv2c(x,h);
    return sol
end

function diffv(x)
    h=[0 1 -1]';
    sol=conv2c(x,h);
    return sol
end

function TVnorm(x)
    y = sum(sum(sqrt.(diffh(x).^2 .+ diffv(x).^2)));
    return y
end