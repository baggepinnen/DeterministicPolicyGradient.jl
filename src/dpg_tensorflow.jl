"""
`DPGopts(m;σβ=1.,αΘ=0.0001,αw=0.001,αu=0.001,γ=0.99,τ=0.001,iters=20_000, critic_update=:gradient,λrls=0.999,stepreduce_interval=1000,stepreduce_factor=0.995,hold_actor=1000)`

Structure with options to the DMP

# Fields

`σβ` Exploration noise covariance\n
`αΘ::Float64` Actor step size\n
`αw::Float64` Q-function step size 1\n
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
    γ::Float64
    τ::Float64
    λ::Float64
    iters::Int64
    m::Int64
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
γ=0.99,
τ=0.001,
λ=1e-4,
iters=20_000,
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
DPGopts(σβ,αΘ,αw,γ,τ,λ,iters,m,stepreduce_interval,stepreduce_factor,hold_actor,experience_replay,experience_ratio,momentum,rmsprop,rmspropfactor,critic_update_interval,eval_interval,divergence_threshold)

"""
Structure with functions to pass to the DMP

# Fields

`μ,Q,gradients,simulate,exploration,reward`

`∇aQ, ∇μ = gradients(s1,s,a1,a,Θ,w,v,t,C)`

See example file or the paper by Silver et al. 2014
"""
immutable DPGfuns
    μ::Function
    Q
    gradients::Function
    simulate::Function
    exploration::Function
    reward::Function
end

"""
Structure which contains the parameters of the DPG optimization problem

`Θ` parameters in the actor
`w` parameters in the Q-function
<!--  -->
"""
type DPGstate{T1<:AbstractVector}
    Θ::T1
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
function dpg(session, train_step, opts, funs, state0, x0, progressfun = (Θ,w,i,x,uout,cost)->0)
    println("=== Deterministic Policy Gradient ===")
    # Expand input structs
    σβ          = opts.σβ
    αΘ          = opts.αΘ
    αw          = opts.αw
    γ           = opts.γ
    τ           = opts.τ
    iters       = opts.iters
    m           = opts.m
    n           = length(x0)
    mem         = SequentialReplayMemory(opts.experience_replay)
    μ           = funs.μ
    Q           = funs.Q
    gradients   = funs.gradients
    simulate    = funs.simulate
    exploration = funs.exploration
    r           = funs.reward

    # Initialize parameters
    Θ           = deepcopy(state0.Θ) # Weights
    Θt          = deepcopy(Θ) # Tracking weights
    w = [0]
    wt          = deepcopy(w)
    Pw          = size(Θ,1)
    Θb          = deepcopy(Θ) # Best weights
    wb          = deepcopy(w)
    dΘs         = 1000ones(Pw) # Weight gradient states
    dΘs2        = 0*ones(Pw)
    dws         = 100ones(Pw)
    cost        = zeros(iters)
    bestcost    = Inf
    x,uout      = simulate(Θ, x0)
    T           = size(x,1)

    s = zeros(n)

    function train!(rollout, update_actor)
        dΘ = handle_rollout(rollout)
        train_critic()
        actor_update!(Θ,αΘ,dΘ,dΘs,dΘs2,update_actor,opts)
        update_tracking_networks()
        nothing
    end


    # ====== Helper functions =========================

    function train_critic()
        # Update the critic network using experience replay
        # Maybe take several steps here, i.e., sample many mini batches
        batch = sample_uniform!(mem,batch_size)
        for (it,t) in enumerate(batch)
            a1          = μ(t.s1,Θ,t.t)
            q           = run(session, Q,  Dict(s_pl => t.s1, a_pl => a1))[2] # Tracking, I think
            y           = t.r + γ * q
            targets[it] = y
        end
        run(session, train_step, Dict(s_pl => s, a_pl => a, y_pl => targets))
    end

    function experience_replay(i)
        for ei = 1:opts.experience_ratio
            trans = sample_uniform!(mem,T-1)
            train!(trans, i > opts.hold_actor)
            push!(mem, trans)
        end
        ((i % 50) == 0) && sort!(mem)
    end

    function handle_rollout(rollout)
        dΘ = zeros(Θ)
        for t in rollout
            a1          = μ(t.s1,Θt,t.t)
            ∇aQ, ∇μ     = gradients(t.s1,t.s,a1,t.a,Θ,w,t.t)
            q1           = run(session, Q,  Dict(s => t.s1, a => a1))[2] # Tracking
            y           = t.r + γ * q1
            q           = run(session, Q,  Dict(s => t.s, a => a))[2] # Not tracking
            t.δ         = (y - q)[1]
            dΘ += ∇μ*∇aQ
        end
        dΘ./= length(rollout)
    end

    function create_rollout(x,uout,i)
        rollout     = Vector{Transition}(T-1)
        for ti = 1:T-1
            s1          = x[ti+1,:][:]
            s           = x[ti,:][:]
            a           = uout[ti,:][:]
            ri          = r(s1,a,ti)
            cost[i]    -= ri
            trans       = Transition(s,s1,a,ri,0.,ti,i)
            rollout[ti] = trans
            opts.experience_replay > 0 && push!(mem, trans)
        end
        rollout
    end

    function update_tracking_networks()
        Θt[:], wt[:] = τ*Θ + (1-τ)*Θt, τ*w + (1-τ)*wt
    end

    function periodic_evaluation(i)
        x,uout = simulate(Θ, x0) # Evaluate using latest parameters and possibly revert back to tracking parameters
        cost[i] = J(x,uout,r)
        progressfun(Θ,w,i,x,uout, cost)
        println(i, ", cost: ", cost[i] |> r5, " norm ∇Θ: ", Σ½(dΘs) |> r5)

        if cost[i] < bestcost
            bestcost = cost[i]
            # TODO: changed to saveing tracking networks, to be more likely to escape local minima
            Θb = deepcopy(Θ)
            wb = deepcopy(w)

        elseif cost[i] > opts.divergence_threshold*bestcost
            αΘ,αw,σβ = reduce_stepsize_divergence!(αΘ,αw,Θb,wb)
        end
    end

    # Main loop ================================================================
    for i = 1:iters
        x0i         = x0 #+ 2randn(n) # TODO: this should not be hard coded
        noise       = exploration(σβ)
        x,uout      = simulate(Θ, x0i, noise)
        rollout     = create_rollout(x,uout,i)
        train!(rollout, i > opts.hold_actor)
        αΘ,αw = reduce_stepsize_periodic!(i,αΘ,αw,opts)
        if opts.experience_replay > 0 && i > 10
            experience_replay(i)
        end
        if (i-1) % opts.eval_interval == 0 # Simulate without noise and evaluate cost
            periodic_evaluation(i)
        end


    end
    println("Done. Minimum cost: $(minimum(cost[1:opts.eval_interval:end])), ($(minimum(cost)))")
    return cost, Θb, wb, mem # Select the parameters with lowest cost
end


function actor_update!(Θ,αΘ,dΘ,dΘs,dΘs2,update_actor,opts)
    dΘs[:] = opts.rmspropfactor*dΘs + (1-opts.rmspropfactor)*dΘ.^2
    if update_actor
        ΔΘ = αΘ * dΘ
        if opts.rmsprop
            ΔΘ ./= sqrt(dΘs+1e-10) # RMSprop
        end
        # ΔΘ = dΘ./sqrt(dΘs+1e-10).*sqrt(dΘs2) # ADAdelta
        # dΘs2[:] = 0.9dΘs2 + 0.1ΔΘ.^2 # ADAdelta
        dΘs2[:] = opts.momentum*dΘs2 + ΔΘ # Momentum + RMSProp
        Θ[:]  += dΘs2
    end
end


function critic_update_gradient!(w,T,αw,dw,dws,opts)
    Δw = αw/T * dw
    dws[:] = opts.rmspropfactor*dws + (1-opts.rmspropfactor)*dw.^2
    if opts.rmsprop
        Δw ./= sqrt(dws)+1e-10
    end
    w += Δw
end


function reduce_stepsize_periodic!(i,αΘ,αw,opts)
    if i % opts.stepreduce_interval != 0
        return αΘ,αw
    end
    αΘ  *= opts.stepreduce_factor
    αw  *= opts.stepreduce_factor
    αΘ,αw
end

function reduce_stepsize_divergence!(αΘ,αw,Θb,wb)
    print_with_color(:orange,"Reducing stepsizes due to divergence\n")
    αΘ  /= 5
    αw  /= 5
    σβ  /= 2
    Θ[:], w[:] = deepcopy(Θb), deepcopy(wb) # reset parameters
    αΘ,αw,σβ
end
