"""
`DPGopts(m;σβ=1.,αΘ=0.0001,αw=0.001,αv=0.001,αu=0.001,γ=0.99,τ=0.001,iters=20_000, critic_update=:gradient,λrls=0.999,stepreduce_interval=1000,stepreduce_factor=0.995,hold_actor=1000)`

Structure with options to the DMP

# Fields

`σβ` Exploration noise covariance\n
`αΘ::Float64` Actor step size\n
`αw::Float64` Q-function step size 1\n
`αv::Float64` Q-function step size 2\n
`αu::Float64` Currently not used\n
`γ::Float64` Discount factor\n
`τ::Float64` Tracking factor between target and training networks\n
`λ::Float64` Regularization parameter for leastsquares Q-function fitting\n
`iters::Int64` Number of iterations to run\n
`m::Int64` Action dimension\n
`critic_update::Symbol` How to update the critic, can be chosen as `:gradient`, `:rls`, `:kalman`, `:leastsquares`\n
`λrls::Float64` If rls is used for critic update, use this forgetting factor\n
`stepreduce_interval::Int` The stepsize is reduced with this interval\n
`stepreduce_factor::Float64` The stepsize is reduced with this factor\n
`hold_actor::Int` Keep the actor from being updated for a few iterations in the beginning to allow the critic to obtain reasonable values\n
`experience_replay::Int` How many transitions to store in the replay buffer\n
`experience_ratio::Int` How many stored transitions to train with for every new experience\n
`momentum::Float64` Momentum term in actor gradient ascent (0.9)\n
`rmsprop::Bool` Use rmsprop for gradient training\n
`rmspropfactor::Float64` How much rmsprop, (0.9)\n
`critic_update_interval::Int` If critic is trained using leastsquares, only relearn the critic between this many mini-batches\n
`eval_interval::Int` With this interval the policy is evaluated without noise, parameters saved and stepsizes are reduced if divergence is detected.\n
`divergence_threshold::Float64` If the cost is higher than `divergence_threshold*bestcost`, the parameters are reset to their previously best values and stepsizes are reduced.\n
See example file or the paper Deterministic Policy Gradient Algorithms, Silver et al. 2014
"""
immutable DPGopts
    σβ
    αΘ::Float64
    αw::Float64
    αv::Float64
    αu::Float64
    γ::Float64
    τ::Float64
    λ::Float64
    iters::Int64
    m::Int64
    critic_update::Symbol
    λrls::Float64
    stepreduce_interval::Int
    stepreduce_factor::Float64
    hold_actor::Int
    experience_replay::Int
    experience_ratio::Int
    momentum::Float64
    rmsprop::Bool
    rmspropfactor::Float64
    critic_update_interval::Int
    eval_interval::Int
    divergence_threshold::Float64
end

DPGopts(m;
σβ=1.,
αΘ=0.001,
αw=0.01,
αv=0.01,
αu=0.001,
γ=0.99,
τ=0.001,
λ=1e-4,
iters=20_000,
critic_update=:gradient,
λrls=0.999,
stepreduce_interval=1000,
stepreduce_factor=0.995,
hold_actor=1000,
experience_replay=0,
experience_ratio=10,
momentum=0.9,
rmsprop=true,
rmspropfactor=0.9,
critic_update_interval=1,
eval_interval=10,
divergence_threshold=1.5) =
DPGopts(σβ,αΘ,αw,αv,αu,γ,τ,λ,iters,m,critic_update,λrls,stepreduce_interval,stepreduce_factor,hold_actor,experience_replay,experience_ratio,momentum,rmsprop,rmspropfactor,critic_update_interval,eval_interval,divergence_threshold)

"""
Structure with functions to pass to the DMP

# Fields

`μ,Q,gradients,simulate,exploration,reward`

See example file or the paper by Silver et al. 2014
"""
immutable DPGfuns
    μ::Function
    Q::Function
    gradients::Function
    simulate::Function
    exploration::Function
    reward::Function
end

"""
Structure which contains the parameters of the DPG optimization problem\n
`Θ` parameters in the actor\n
`w` parameters in the Q-function\n
`v` parameters in the Q-function\n
All parameters should be a subtype of AbstractVector
A typical Q-function looks like `Q = (∇μ(s)*(a-μ(s)))'w + V(s,v)`

"""
type DPGstate{T1<:AbstractVector,T2<:AbstractVector,T3<:AbstractVector}
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
# `cost, Θ, w, v = dpg(opts, funs, state0, x0)`
#
# Main function.
#
# # Arguments
# `opts::DPGopts` structure with options and parameters\n
# `funs::DPGfuns` structure with functions\n
# `state0::DPGstate` initial parameters
# `x0` initial system state
# """
function dpg(opts, funs, state0, x0,C, progressfun = (Θ,w,v,C,i,x,uout,cost)->0)
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
    n           = length(x0)
    critic_update= opts.critic_update
    λrls        = opts.λrls
    mem         = SequentialReplayMemory(opts.experience_replay)
    μ           = funs.μ
    Q           = funs.Q
    gradients   = funs.gradients
    simulate    = funs.simulate
    exploration = funs.exploration
    r           = funs.reward
    println("Training using $critic_update")

    # Initialize parameters
    Θ           = deepcopy(state0.Θ) # Weights
    w           = deepcopy(state0.w)
    v           = deepcopy(state0.v)
    Θt          = deepcopy(Θ) # Tracking weights
    wt          = deepcopy(w)
    vt          = deepcopy(v)
    Pw          = size(Θ,1)
    Pv          = size(v,1)
    Θb          = deepcopy(Θ) # Best weights
    wb          = deepcopy(w)
    vb          = deepcopy(v)
    dΘs         = 1000ones(Pw) # Weight gradient states
    dΘs2        = 0*ones(Pw)
    dws         = 100ones(Pw)
    dvs         = 100ones(Pv)
    cost        = zeros(iters)
    bestcost    = Inf
    x,uout      = simulate(Θ, x0)
    T           = size(x,1)

    # TODO: Make the parameters below part of the options
    if critic_update == :rls
        Pvw = 10eye(Pw+Pv)
    elseif critic_update == :kalman
        Pk = 1000eye(Pw+Pv)
        R2 = 1
        R12 = zeros(Pw+Pv)
    end

    s = zeros(n)

    function train!(tvec, update_actor,update_critic=true)
        dΘ,dw,dv = zeros(Θ),zeros(w),zeros(v)
        if critic_update ==:leastsquares && update_critic
            basis = sample_uniform!(mem,Pv)
            for (it,t) in enumerate(basis)
                C[:,it] = t.s
            end
            push!(mem,basis)
            A = Matrix{Float64}(length(mem),Pv+Pw)
            targets = Vector{Float64}(length(mem))
            for (it,t) in enumerate(mem)
                a1          = μ(t.s1,Θ,t.t)
                _, ∇wQ, ∇vQ, _ = gradients(t.s1,t.s,a1,t.a,Θ,w,v,t.t,C)
                y           = t.r + γ * Q(t.s1,a1,vt,wt,Θt,t.t,C)
                targets[it] = y
                # t.δ         = (y - Q(t.s,t.a,v,w,Θ,t.t))[1] # Not needed for batch learning
                A[it,1:Pv]  = ∇vQ
                A[it,Pv+1:end]  = ∇wQ
            end
            vw = [A; opts.λ*eye(Pv+Pw)]\[targets+100;zeros(Pv+Pw)] #TODO magic numer 100?
            v[:],w[:] = vw[1:Pv],vw[Pv+1:end]
        end
        for t in tvec
            a1          = μ(t.s1,Θt,t.t)
            ∇aQ, ∇wQ,∇vQ, ∇μ = gradients(t.s1,t.s,a1,t.a,Θ,w,v,t.t,C)
            y           = t.r + γ * Q(t.s1,a1,vt,wt,Θt,t.t,C)
            t.δ         = (y - Q(t.s,t.a,v,w,Θ,t.t,C))[1]
            if critic_update == :rls
                vw,Pvw      = RLS([v;w], y, [∇vQ;∇wQ], Pvw, λrls)
                v[:],w[:]   = vw[1:Pv],vw[Pv+1:end]
            elseif critic_update == :kalman

                Φ = [∇vQ;∇wQ]
                # R1 = ΦΦ', to only update covariance in the direction of incoming data
                vw,Pk = kalman(Φ*Φ',R2,R12,[v;w], y, Φ, Pk)
                v[:],w[:] = vw[1:Pv],vw[Pv+1:end]
            elseif critic_update ==:gradient
                dw += t.δ * ∇wQ  #- γ * ϕ(s1,a1) * ϕu
                dv += t.δ * ∇vQ   #- γ * ϕ(s1) * ϕu
            end
            dΘ += ∇μ*∇aQ
        end
        dΘ./= length(tvec)
        # RMS prop update parameters (gradient divided by running average of RMS gradient, see. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        dΘs[:] = opts.rmspropfactor*dΘs + (1-opts.rmspropfactor)*dΘ.^2
        if update_actor
            ΔΘ = αΘ * dΘ
            if opts.rmsprop
                ΔΘ ./= sqrt(dΘs+1e-10) # RMSprop
            end
            # ΔΘ = dΘ./sqrt(dΘs+1e-10).*sqrt(dΘs2) # ADAdelta
            # dΘs2[:] = 0.9dΘs2 + 0.1ΔΘ.^2 # ADAdelta
            dΘs2 = opts.momentum*dΘs2 + ΔΘ # Momentum + RMSProp
            Θ[:]  += dΘs2
        end

        if critic_update == :gradient
            Δw = αw/T * dw
            Δv = αv/T * dv
            dws[:] = opts.rmspropfactor*dws + (1-opts.rmspropfactor)*dw.^2
            dvs[:] = opts.rmspropfactor*dvs + (1-opts.rmspropfactor)*dv.^2
            if opts.rmsprop
                Δw ./= sqrt(dws)+1e-10
                Δv ./= sqrt(dvs)+1e-10
            end
            w += Δw
            v += Δv
        end

        # Update tracking networks
        Θt[:], wt[:], vt[:] = τ*Θ + (1-τ)*Θt, τ*w + (1-τ)*wt, τ*v + (1-τ)*vt
        nothing
    end

    # Main loop ================================================================
    for i = 1:iters
        x0i         = x0 #+ 2randn(n) # TODO: this should not be hard coded
        noise       = exploration(σβ)
        x,uout      = simulate(Θ, x0i, noise)
        batch = Vector{Transition}(T-1)
        for ti = 1:T-1
            s1          = x[ti+1,:][:]
            s           = x[ti,:][:]
            a           = uout[ti,:][:]
            ri          = r(s1,a,ti)
            cost[i]    -= ri
            trans       = Transition(s,s1,a,ri,0.,ti,i)
            batch[ti]   = trans
            opts.experience_replay > 0 && push!(mem, trans)
        end
        train!(batch, i > opts.hold_actor)

        if i % opts.stepreduce_interval == 0
            αΘ  *= opts.stepreduce_factor
            αw  *= opts.stepreduce_factor
            αv  *= opts.stepreduce_factor
        end

        if opts.experience_replay > 0 && i > 10
            for ei = 1:opts.experience_ratio
                trans = sample_uniform!(mem,T-1)
                train!(trans, i > opts.hold_actor, ei-1 % opts.critic_update_interval == 0)
                push!(mem, trans)
            end
            ((i % 50) == 0) && sort!(mem)
        end

        if (i-1) % opts.eval_interval == 0 # Simulate without noise and evaluate cost
            x,uout = simulate(Θ, x0) # Evaluate using latest parameters and possibly revert back to tracking parameters
            cost[i] = J(x,uout,r)
            progressfun(Θ,w,v,C,i,x,uout, cost)
            if critic_update == :gradient
                println(i, ", cost: ", cost[i] |> r5, " norm ∇Θ: ", Σ½(dΘs) |> r5, " norm ∇w: ", Σ½(dws) |> r5, " norm ∇v: ", Σ½(dvs) |> r5)#, " trace(P): ", trace(Pvw) |> r5)
            else
                println(i, ", cost: ", cost[i] |> r5, " norm ∇Θ: ", Σ½(dΘs) |> r5)
            end
            if cost[i] < bestcost
                bestcost = cost[i]
                # TODO: changed to saveing tracking networks, to be more likely to escape local minima
                Θb = deepcopy(Θ)
                wb = deepcopy(w)
                vb = deepcopy(v)

            elseif cost[i] > opts.divergence_threshold*bestcost
                print_with_color(:orange,"Reducing stepsizes due to divergence\n")
                αΘ  /= 5
                αw  /= 5
                αv  /= 5
                αu  /= 5
                σβ  /= 2
                Θ, w, v = deepcopy(Θb), deepcopy(wb), deepcopy(vb) # reset parameters
            end
        end


    end
    println("Done. Minimum cost: $(minimum(cost[1:opts.eval_interval:end])), ($(minimum(cost)))")
    return cost, Θb, wb, vb, mem # Select the parameters with lowest cost
end
