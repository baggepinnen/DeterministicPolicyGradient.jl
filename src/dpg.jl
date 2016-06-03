"""
`DPGopts(m;σβ=1.,αΘ=0.0001,αw=0.001,αv=0.001,αu=0.001,γ=0.99,τ=0.001,iters=20_000, critic_update=:gradient,λrls=0.999,stepreduce_interval=1000,stepreduce_factor=0.995,hold_actor=1000)`

Structure with options to the DMP

# Fields
`σβ, αΘ, αw, αv, αu, γ, τ, iters, m, critic_update, λrls, stepreduce_interval, stepreduce_factor, hold_actor`\n
The `cirtic_update` can be chosen as `:gradient`, `:rls`, `:kalman`
See example file or the paper by Ijspeert et al. 2013
"""
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
    critic_update::Symbol
    λrls::Float64
    stepreduce_interval::Int
    stepreduce_factor::Float64
    hold_actor::Int
end

DPGopts(m;σβ=1.,αΘ=0.0001,αw=0.001,αv=0.001,αu=0.001,γ=0.99,τ=0.001,iters=20_000, critic_update=:gradient,λrls=0.999,stepreduce_interval=1000,stepreduce_factor=0.995,hold_actor=1000) =
DPGopts(σβ,αΘ,αw,αv,αu,γ,τ,iters,m,critic_update,λrls,stepreduce_interval,stepreduce_factor,hold_actor)

"""
Structure with functions to pass to the DMP

# Fields

`μ,Q,gradients,simulate,exploration,reward`

See example file or the paper by Silver et al. 2014
"""
type DPGfuns
    μ::Function
    Q::Function
    gradients::Function
    simulate::Function
    exploration::Function
    reward::Function
end

type DPGstate{T1,T2,T3}
    Θ::T1
    w::T2
    v::T3
end

function J(x,a,r)
    cost = @parallel (+) for t = 1:size(x,1)
    r(x[t,:][:],a[t,:][:],t)
end
-cost
end


# """
# `cost, Θ, w, v = dpg(opts, funs, x0)`
#
# Main function.
#
# # Arguments
# `opts::DPGopts` structure with options and parameters\n
# `funs::DPGfuns` structure with functions\n
# `x0` initial state
# """
function dpg(opts, funs, state0, x0)
    println("=== Deterministic Policy Gradient ===")
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
    n = length(x0)
    critic_update= opts.critic_update
    λrls        = opts.λrls
    μ           = funs.μ
    Q           = funs.Q
    gradients   = funs.gradients
    simulate    = funs.simulate
    exploration = funs.exploration
    r           = funs.reward
    println("Training using $critic_update")

    # Initialize parameters
    Θ           = state0.Θ # Weights
    w           = state0.w
    v           = state0.v
    Θt          = deepcopy(Θ) # Tracking weights
    wt          = deepcopy(w)
    vt          = deepcopy(v)
    Pw          = size(Θ,1)
    Pv          = size(v,1)
    u           = zeros(Pw)
    Θb          = deepcopy(Θ) # Best weights
    wb          = deepcopy(w)
    vb          = deepcopy(v)
    dΘs         = 1000ones(Pw) # Weight gradient states
    dws         = 100ones(Pw)
    dvs         = 100ones(Pw)
    cost        = zeros(iters)
    bestcost    = Inf

    # TODO: Make the parameters below part of the options
    if critic_update == :rls
        Pvw = 0.1eye(Pw+Pv)
    elseif critic_update == :kalman
        Pk = 10000eye(Pw+Pv)
        R2 = 1
        R12 = 0.0ones(Pw+Pv)
    end

    s = zeros(n)

    for i = 1:iters
        x0i         = x0 + 2randn(n) # TODO: this should not be hard coded
        noise       = exploration(σβ)
        x,uout      = simulate(Θ, x0i, noise)
        T           = size(x,1)
        dΘ          = zeros(Θ)
        if critic_update == :gradient
            dw          = zeros(w)
            dv          = zeros(v)
        end
        for ti = 1:T-1
            s1          = x[ti+1,:][:]
            s           = x[ti,:][:]
            a           = uout[ti,:][:]
            a1          = μ(s1,Θ,ti)
            ri          = r(s1,a,ti)
            cost[i]    -= ri
            ∇aQ, ∇wQ,∇vQ, ∇μ = gradients(s1,s,a1,a,Θ,w,v,ti)
            dΘ         += ∇μ*∇aQ
            y           = ri + γ * Q(s1,a1,vt,wt,Θt,ti)
            if critic_update == :rls
                vw,Pvw  = RLS([v;w], y, [∇vQ;∇wQ], Pvw, λrls)
                v,w     = vw[1:Pv],vw[Pv+1:end]
            elseif critic_update == :kalman
                Φ = [∇vQ;∇wQ]
                # R1 = ΦΦ', to only update covariance in the direction of incoming data
                vw,Pk = kalman(Φ*Φ',R2,R12,[v;w], y, Φ, Pk)
                v,w = vw[1:Pv],vw[Pv+1:end]
            else
                δ           = (y - Q(s,a,v,w,Θ,ti))[1]
                dw         += δ * ∇wQ  #- γ * ϕ(s1,a1) * ϕu
                dv         += δ * ∇vQ   #- γ * ϕ(s1) * ϕu
            end

        end

        # RMS prop update parameters (gradient divided by running average of RMS gradient, see. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf slide 29
        if i > opts.hold_actor
            dΘs = 0.9dΘs + 0.1dΘ.^2
            Θ = Θ + αΘ/T * dΘ./(sqrt(dΘs)+0.00001)
        end
        if critic_update == :gradient
            dws = 0.9dws + 0.1dw.^2
            dvs = 0.9dvs + 0.1dv.^2
            w = w + αw/T * dw./(sqrt(dws)+0.000001)
            v = v + αv/T * dv./(sqrt(dvs)+0.000001)
        end

        # Update tracking networks
        Θt, wt, vt = τ*Θ + (1-τ)*Θt, τ*w + (1-τ)*wt, τ*v + (1-τ)*vt

        if i % opts.stepreduce_interval == 0
            αΘ  *= opts.stepreduce_factor
            αw  *= opts.stepreduce_factor
            αv  *= opts.stepreduce_factor
        end

        if (i-1) % 100 == 0 # Simulate without noise and evaluate cost # TODO: remove hard coded 100
            x,uout = simulate(Θ, x0)
            cost[i] = J(x,uout,r)
            if critic_update == :gradient
                println(i, ", cost: ", cost[i] |> r5, " norm ∇Θ: ", Σ½(dΘs) |> r5, " norm ∇w: ", Σ½(dws) |> r5, " norm ∇v: ", Σ½(dvs) |> r5)#, " trace(P): ", trace(Pvw) |> r5)
            else
                println(i, ", cost: ", cost[i] |> r5, " norm ∇Θ: ", Σ½(dΘs) |> r5)
            end
            if cost[i] < bestcost
                bestcost = cost[i]
                Θb = deepcopy(Θ)
                wb = deepcopy(w)
                vb = deepcopy(v)
            elseif cost[i] > 1.2bestcost
                print_with_color(:orange,"Reducing stepsizes due to divergence")
                αΘ  /= 10
                αw  /= 10
                αv  /= 10
                αu  /= 10
                σβ ./= 2
                Θ, w, v = deepcopy(Θb), deepcopy(wb), deepcopy(vb) # reset parameters
            end
        end

    end
    println("Done. Minimum cost: $(minimum(cost[1:100:end])), ($(minimum(cost)))")
    return cost, Θb, wb, vb # Select the parameters with lowest cost
end
