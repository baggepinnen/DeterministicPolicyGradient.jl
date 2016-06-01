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
