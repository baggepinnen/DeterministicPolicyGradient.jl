r5(x) = round(x,5)
Σ½(x) = sqrt(sum(x))

function meshgrid(a,b)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end

function meshgrid2(a,b,c)
    grid_a = [i for i in a, j in b, k in c]
    grid_b = [j for i in a, j in b, k in c]
    grid_c = [k for i in a, j in b, k in c]
    grid_a, grid_b, grid_c
end

"""
`C, Σ = get_centers_multi(centers,σvec)`\n
Returns multidimensional centers and covariance matrices by gridding.
´centers´ is a matrix with n_basis×n_signals center coordinates\n
`σvec` is a matrix with the corresponding n_basis×n_signals widths
`C, Σ` are the corresponding multidimensional center locations and shapes, both are n_signals × nbasis^n_signals
"""
function get_centers_multi(centers,σvec)
    (nbasis,Nsignals) = size(centers)
    Nbasis::Int64 = nbasis^Nsignals
    Centers = zeros(Nsignals, Nbasis)
    Σ = zeros(Nsignals, Nbasis)
    v = Nbasis
    h = 1
    for i = 1:Nsignals
        v = convert(Int64,v / nbasis)
        Centers[i,:] = vec(repmat(centers[:,i]',v,h))'
        Σ[i,:] = vec(repmat(σvec[:,i]',v,h))'
        h *= nbasis
    end
    Centers, Σ
end

function RLS!(Θ, y, ϕ, P, λ)
    Pϕ = P*ϕ
    P[:,:] = 1/λ*(P - (Pϕ*Pϕ')./(λ + ϕ'*Pϕ))
    yp = (ϕ'Θ)[1]
    e = y-yp
    Θ += Pϕ*e
    return nothing
end

function RLS(Θ, y, ϕ, P, λ)
    Pϕ = P*ϕ
    P = 1/λ*(P - (Pϕ*Pϕ')./(λ + ϕ'*Pϕ))
    yp = (ϕ'Θ)[1]
    e = y-yp
    Θ = Θ + Pϕ*e
    return Θ, P
end

function kalman(R1,R2,R12,Θ, y, ϕ, P)
    ϕTP = ϕ'P
    K = (P*ϕ+R12)/(R2+ϕTP*ϕ)
    P = P - (P*ϕ+R12)/(R2 + ϕTP*ϕ)*(ϕTP+R12') + R1
    yp = (ϕ'Θ)[1]
    e = y-yp
    Θ = Θ + K*e
    return Θ, P
end


function quadform(a,Q)
    vecdot(a,(Q*a))
end

function batch2sat{T}(batch::Batch{T}, μ, Θ)
    batch_size = length(batch)
    s = Matrix{T}(batch_size,length(batch[1].s))
    a = Matrix{T}(batch_size,length(batch[1].a))
    ts = Vector{T}(batch_size)
    for (i,trans) in enumerate(batch)
        s[i,:]     = trans.s
        a[i,:]     = μ(trans.s,Θ, trans.t)
        ts[i]      = trans.t
    end
    s,a,ts
end

function batch2say{T}(batch::Batch{T}, Q, μ, Θ, γ)
    batch_size = length(batch)
    s = Matrix{T}(batch_size,length(batch[1].s))
    a = Matrix{T}(batch_size,length(batch[1].a))
    y = Vector{T}(batch_size)
    for (i,trans) in enumerate(batch)
        a1     = μ(trans.s1,Θ,trans.t)
        q      = Q(trans.s1,a1,trans.t)
        y[i]   = trans.r + γ * q
        s[i,:] = trans.s
        a[i,:] = trans.a
    end
    s,a,y
end

function batch2say_montecarlo{T}(batch::Batch{T}, γ)
    batch_size = length(batch)
    s = Matrix{T}(batch_size,length(batch[1].s))
    a = Matrix{T}(batch_size,length(batch[1].a))
    r = Vector{T}(batch_size)
    for (i,trans) in enumerate(batch)
        r[i]   = trans.r
        s[i,:] = trans.s
        a[i,:] = trans.a
    end
    y = discounted_return(r,γ)
    s,a,y
end

function batch2all{T}(batch::Batch{T}, Q, μ, Θ, γ)
    batch_size = length(batch)
    s   = Matrix{T}(batch_size,length(batch[1].s))
    s1  = Matrix{T}(batch_size,length(batch[1].s))
    a   = Matrix{T}(batch_size,length(batch[1].a))
    a1  = Matrix{T}(batch_size,length(batch[1].a))
    y   = Vector{T}(batch_size)
    q   = Vector{T}(batch_size)
    r   = Vector{T}(batch_size)
    ts  = Vector{T}(batch_size)
    for (i,trans) in enumerate(batch)
        a1i     = μ(trans.s1,Θ,trans.t)
        a1[i,:] = a1i
        q1      = Q(trans.s1,a1i,trans.t)
        q[i]    = Q(trans.s,trans.a,trans.t)
        y[i]    = trans.r + γ * q1
        r[i]    = trans.r
        s[i,:]  = trans.s
        s1[i,:] = trans.s1
        a[i,:]  = trans.a
        ts[i]   = trans.t
    end
    s,s1,a,a1,q,y,r,ts
end

function discounted_return(r,γ)
    @assert (0.9 <= γ <= 1) "Gamma has a weird value in discounted_return: γ = $γ"
    l = length(r)
    retur = similar(r)
    retur[l] = r[l]
    for i = l-1:-1:1
        retur[i] = r[i] + γ*retur[i+1]
    end
    retur
end


function Qplot!{T}(batch::Batch{T}, Q, μ, Θ, γ; kwargs...)
    batch_size = length(batch)
    q   = Vector{T}(batch_size)
    r   = Vector{T}(batch_size)
    for (i,trans) in enumerate(batch)
        # a1     = μ(trans.s1,Θ,trans.t)
        # q1      = Q(trans.s1,a1,trans.t)
        q[i]    = Q(trans.s,trans.a,trans.t)
        r[i]    = trans.r
    end
    ret = discounted_return(r,γ)
    mir,mar = extrema(ret)
    plot!([mir,mar],[mir,mar]; lab = "", kwargs...)
    scatter!(ret,q; ylabel="Estimated Q", xlabel="Actual return",markersize=3, markerstrokealpha=0, kwargs...)
end
function Qplot{T}(batch::Batch{T}, Q, μ, Θ, γ; kwargs...)
    f = plot()
    Qplot!(batch, Q, μ, Θ, γ; kwargs...)
    f
end

function autoscale(T::Type,s...)
    d = s[1]
    return 4/√(d)*rand(T,s...) - 2/√(d)
end
autoscale(s...) = autoscale(Float32,s...)

function update_plot!(p; max_history = 10, attribute = :markercolor)
    num_series = length(p.series_list)
    if num_series > 1
        if num_series > max_history
            deleteat!(p.series_list,1:num_series-max_history)
        end
        for i = 1:min(max_history, num_series)-1
            alpha = 1-2/max_history
            c = p[i][attribute]
            b = alpha*c.b + (1-alpha)*0.5
            g = alpha*c.g + (1-alpha)*0.5
            r = alpha*c.r + (1-alpha)*0.5
            a = alpha*c.alpha
            p[i][attribute] = RGBA(r,g,b,a)
        end
    end

end
