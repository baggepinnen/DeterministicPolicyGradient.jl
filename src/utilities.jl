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

function RLS!(Θ, y, ϕ, P, λ)
    Pϕ = P*ϕ
    P = 1/λ*(P - (Pϕ*Pϕ')./(λ + ϕ'*Pϕ))
    yp = (ϕ'Θ)[1]
    e = y-yp
    Θ += Pϕ*e
    return P
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
