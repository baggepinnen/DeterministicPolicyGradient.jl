
type DPGopts
    σβ
    αΘ::Float64
    αw::Float64
    αv::Float64
    αu::Float64
    γ::Float64
    τ::Float64
    iters::Int64
    m::Int64
    rls_critic::Bool
    λrls::Float64
end

DPGopts(m;σβ=1.,αΘ=0.0001,αw=0.001,αv=0.001,αu=0.001,γ=0.99,τ=0.001,iters=20_000, rls_critic=true,λrls=0.999) =
DPGopts(σβ,αΘ,αw,αv,αu,γ,τ,iters,m,rls_critic,λrls)

type DPGfuns
    μ::Function
    ∇μ::Function
    β::Function
    ϕ::Function
    V::Function
    Q::Function
    simulate::Function
    exploration::Function
end

function J(x,a,r)
    cost = @parallel (+) for t = 1:size(x,1)
    r(x[t,:][:],a[t])
end



function dpg(opts, funs, x0)
    # Expand input structs
    σβ          = opts.σβ
    αΘ          = opts.αΘ
    αw          = opts.αw
    αv          = opts.αv
    αu          = opts.αu
    γ           = opts.γ
    τ           = opts.τ
    iters       = opts.iters
    m           = opts.m
    rls_critic  = opts.rls_critic
    λrls        = opts.λrls
    μ           = funs.μ
    ∇μ          = funs.∇μ
    β           = funs.β
    ϕ           = funs.ϕ
    V           = funs.V
    Q           = funs.Q
    simulate    = funs.simulate
    exploration = funs.exploration

    # Determine sizes
    n = length(x0)
    P = length(ϕ(x0))


    # Initialize parameters
    Θ           = zeros(P,m) # Weights
    w           = 0.001randn(P)
    v           = 0.001randn(P)
    Θt          = deepcopy(Θ) # Tracking weights
    wt          = deepcopy(w)
    vt          = deepcopy(v)
    u           = zeros(P)
    Θb          = deepcopy(Θ) # Best weights
    wb          = deepcopy(w)
    vb          = deepcopy(v)
    dΘs         = 1000ones(P,m) # Weight gradient states
    dws         = 100ones(P)
    dvs         = 100ones(P)
    cost        = zeros(iters)
    bestcost    = Inf
    if rls_critic
        Pw = 1eye(P)
        Pv = 1eye(P)
        Pvw = 0.1eye(2P)
        Pk = 10000eye(2P)
        R2 = 1
        R12 = 0.0ones(2P)
    end

    s = zeros(n)
    # ∇μ = ForwardDiff.gradient(Θ -> μ(s,Θ))


    for i = 1:iters
        x0i         = x0 + 2randn(n)
        noise       = exploration(σβ)
        x,uout      = simulate(Θ, x0i, noise)
        T           = size(x,1)
        dΘ          = zeros(Θ)
        if !rls_critic
            dw          = zeros(w)
            dv          = zeros(v)
        end
        for ti = 1:T-1
            s1          = x[ti+1,:][:]
            s           = x[ti,:][:]
            a           = uout[ti,:][:]
            a1          = μ(s1,Θ)
            ri          = r(s1,a) # TODO: figure out if s1 or s goes here, probably s1
            cost[i]    -= ri

            ∇i          = ∇μ(s)
            ϕi          = ϕ(s)
            ϕia         = ϕ(s,a,Θ)
            # ϕu          = (ϕia'u)[1]

            dΘ         += ∇i* (∇i'w)[1]
            if rls_critic
                y = ri + γ * Q(s1,a1,vt,wt,Θt)
                # vw,Pvw = RLS([v;w], y, [ϕi;ϕia], Pvw, λrls)
                Φ = [ϕi;ϕia]
                # R1 = ΦΦ', to only update covariance in the direction of incoming data
                vw,Pk = kalman(Φ*Φ',R2,R12,[v;w], y, Φ, Pk)
                v,w = vw[1:P],vw[P+1:end]
            else
                δ           = (ri + γ * Q(s1,a1,vt,wt,Θt) - Q(s,a,v,w,Θ))[1]
                dw         += δ * ϕia  #- γ * ϕ(s1,a1) * ϕu
                dv         += δ * ϕi   #- γ * ϕ(s1) * ϕu
            end
            # u += αu * (δ - ϕu)*ϕia

        end

        # RMS prop update parameters (gradient divided by running average of RMS gradient, see. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf slide 29
        if i > 1_000
            dΘs = 0.9dΘs + 0.1dΘ.^2
            Θ = Θ + αΘ/T * dΘ./(sqrt(dΘs)+0.00001)
        end
        if !rls_critic
            dws = 0.9dws + 0.1dw.^2
            dvs = 0.9dvs + 0.1dv.^2
            w = w + αw/T * dw./(sqrt(dws)+0.000001)
            v = v + αv/T * dv./(sqrt(dvs)+0.000001)
        end

        # Update tracking networks
        Θt, wt, vt = τ*Θ + (1-τ)*Θt, τ*w + (1-τ)*wt, τ*v + (1-τ)*vt

        if i % 5000 == 0
            αΘ  *= 0.995
            αw  *= 0.995
            αv  *= 0.995
        end

        if (i-1) % 100 == 0 # Simulate without noise and evaluate cost
            x,uout = simulate(Θ, x0)
            cost[i] = J(x,uout,r)
            println(i, ", cost: ", cost[i] |> r5, " norm ∇Θ: ", sqrt(sum(dΘs)) |> r5, " norm ∇w: ", sqrt(sum(dws)) |> r5, " norm ∇v: ", sqrt(sum(dvs)) |> r5)#, " trace(P): ", trace(Pvw) |> r5)
            if cost[i] < bestcost
                bestcost = cost[i]
                Θb = deepcopy(Θ)
                wb = deepcopy(w)
                vb = deepcopy(v)
            elseif cost[i] > 1.2bestcost
                αΘ  /= 10
                αw  /= 10
                αv  /= 10
                αu  /= 10
                σβ ./= 2
                Θ, w, v = deepcopy(Θb), deepcopy(wb), deepcopy(vb) # reset parameters
            end
        end

    end

    Θ, w, v = Θb, wb, vb # Select the parameters with lowest cost

    # Plot results


    return cost, Θ, w, v
end
