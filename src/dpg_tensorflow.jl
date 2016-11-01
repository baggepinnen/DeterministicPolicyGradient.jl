"""
`DPGopts(m;αΘ=0.0001,αw=0.001,αu=0.001,γ=0.99,τ=0.001,iters=20_000,stepreduce_interval=1000,stepreduce_factor=0.995,hold_actor=1000)`

Structure with options to the DMP

# Fields

`αΘ::Float64` Actor step size\n
`αw::Float64` Q-function step size 1\n
`αu::Float64` Currently not used\n
`γ::Float64` Discount factor\n
`τ::Float64` Tracking factor between target and training networks\n
`iters::Int64` Number of iterations to run\n
`m::Int64` Action dimension\n
`rls::Float64` If rls is used for critic update, use this forgetting factor\n
`stepreduce_interval::Int` The stepsize is reduced with this interval\n
`stepreduce_factor::Float64` The stepsize is reduced with this factor\n
`hold_actor::Int` Keep the actor from being updated for a few iterations in the beginning to allow the critic to obtain reasonable values\n
`experience_replay::Int` How many transitions to store in the replay buffer\n
`experience_ratio::Int` How many stored transitions to train with for every new experience\n
`momentum::Float64` Momentum term in actor gradient ascent (0.9)\n
`rmsprop::Bool` Use rmsprop for gradient training\n
`rmspropfactor::Float64` How much rmsprop, (0.9)\n
`eval_interval::Int` With this interval the policy is evaluated without noise, parameters saved and stepsizes are reduced if divergence is detected.\n
`divergence_threshold::Float64` If the cost is higher than `divergence_threshold*bestcost`, the parameters are reset to their previously best values and stepsizes are reduced.\n
See example file or the paper Deterministic Policy Gradient Algorithms, Silver et al. 2014
"""
immutable DPGopts
    αΘ::Float64
    αw::Float64
    γ::Float64
    τ::Float64
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
    eval_interval::Int
    divergence_threshold::Float64
end

DPGopts(m;
αΘ=0.001,
αw=0.01,
γ=0.99,
τ=0.001,
iters=20_000,
stepreduce_interval=1000,
stepreduce_factor=0.995,
hold_actor=1000,
experience_replay=0,
experience_ratio=10,
momentum=0.9,
rmsprop=true,
rmspropfactor=0.9,
eval_interval=10,
divergence_threshold=1.5) =
DPGopts(αΘ,αw,γ,τ,iters,m,stepreduce_interval,stepreduce_factor,hold_actor,experience_replay,experience_ratio,momentum,rmsprop,rmspropfactor,eval_interval,divergence_threshold)

"""
Structure with functions to pass to the DMP

# Fields

`μ,Q,gradient,simulate,exploration,reward`

`∇aQ, ∇μ = gradient(s1,s,a1,a,Θ,w,v,t,C)`

See example file or the paper by Silver et al. 2014
"""
immutable DPGfuns
    μ::Function
    Q::Function
    Qt::Function
    gradient::Function
    simulate::Function
    reward::Function
    train_critic::Function
    update_tracking_networks::Function
end

function J(x,a,r)
    cost = -sum(r(x[t+1,:][:],a[t,:][:],t) for t = 1:size(x,1)-1)
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
function dpg(opts, funs, Θ, x0, progressfun = (Θ,i,x,uout,cost, rollout)->0)
    println("=== Deterministic Policy Gradient ===")
    # Expand input structs
    αΘ          = opts.αΘ
    αw          = opts.αw
    γ           = opts.γ
    τ           = opts.τ
    iters       = opts.iters
    m           = opts.m
    n           = length(x0)
    mem         = SequentialReplayMemory(Float32,opts.experience_replay)
    μ           = funs.μ
    Q           = funs.Q
    Qt          = funs.Qt
    gradient   = funs.gradient
    simulate    = funs.simulate

    # Initialize parameters
    Θt          = deepcopy(Θ) # Tracking weights
    Pw          = size(Θ,1)
    Θb          = deepcopy(Θ) # Best weights
    dΘs         = 1e-4ones(Pw) # Weight gradient states
    dΘs2        = 0*ones(Pw)
    dws         = 100ones(Pw)
    cost        = zeros(iters)
    bestcost    = Inf
    x,uout      = simulate(Θ, x0)
    T           = size(x,1)

    s = zeros(n)

    times = zeros(3)

    function train!(rollout, update_actor, montecarlo = false)
        train_critic(rollout, montecarlo)
        if update_actor
            dΘ = policy_gradient(rollout)
            update_actor!(Θ,αΘ,dΘ,dΘs,dΘs2,update_actor,opts)
        end
        # update_tracking_networks(update_actor)
        nothing
    end


    # ====== Helper functions =========================

    function train_critic(batch, montecarlo = false)
        batch_size = length(batch)
        if montecarlo
            s,a,targets = batch2say_montecarlo(batch, γ)
            update_tracking_networks(false) # When we have taken a montecarlo step we update tracking networks immediately so that TD-learning does not pull us back again.
        else
            s,a,targets = batch2say(batch, Qt, μ, Θt, γ)
        end
        funs.train_critic(s,a,targets)
    end

    function experience_replay(i)
        for ei = 1:opts.experience_ratio
            batch = sample_uniform!(mem,T-1)
            # train_critic(batch)
            train!(batch, i > opts.hold_actor)
            push!(mem, batch)
        end
        ((i % 50) == 0) && sort!(mem)
    end

    function policy_gradient(rollout)
        s,aΘ,ts = batch2sat(rollout, μ, Θ)
        dΘ     = gradient(s,aΘ,Θ,ts)
        dΘ   ./= length(rollout)
    end

    function create_rollout(x,uout,i)
        local rollout = Batch{Float32}(T-1)
        for ti = 1:T-1
            s1          = x[ti+1,:][:]
            s           = x[ti,:][:]
            a           = uout[ti,:][:]
            ri          = funs.reward(s1,a,ti)
            trans       = Transition{Float32}(s,s1,a,ri,0.,ti,i)
            rollout[ti] = trans
        end
        rollout
    end

    function update_tracking_networks(update_actor)
        # if opts.experience_ratio == 0 || opts.experience_replay == 0
        #     return
        # end
        funs.update_tracking_networks()
        if update_actor
            Θt[:] = τ*Θ + (1-τ)*Θt
        end
    end

    function periodic_evaluation(i)
        x,uout = simulate(Θ, x0) # Evaluate using latest parameters and possibly revert back to tracking parameters

        cost[i] = J(x,uout,funs.reward)
        local rollout = create_rollout(x,uout,i)
        progressfun(Θ,i,x,uout, cost,rollout)
        println(i, ", cost: ", cost[i] |> r5, " norm ∇Θ: ", Σ½(dΘs) |> r5)
        train!(rollout, i > opts.hold_actor, false) # Here we can actually do MonteCarlo since this is on policy (not on tracking policy though)
        opts.experience_replay > 0 && push!(mem,rollout)
        if cost[i] < bestcost
            bestcost = cost[i]
            Θb = deepcopy(Θ)
        elseif cost[i] > opts.divergence_threshold*bestcost
            αΘ,αw = reduce_stepsize_divergence!(Θ,αΘ,αw,Θb)
        end
    end

    # Main loop ================================================================
    for i = 1:iters
        x0i         = x0 #+ 2randn(n) # TODO: this should not be hard coded
        x,uout      = simulate(Θ, x0i, true)
        cost[i]     = J(x,uout,funs.reward)
        rollout     = create_rollout(x,uout,i)
        update_actor = i > opts.hold_actor
        train!(rollout, update_actor, false)
        update_tracking_networks(update_actor)
        αΘ,αw = reduce_stepsize_periodic!(i,αΘ,αw,opts)
        if opts.experience_replay > 0
            push!(mem, rollout)
            i > 3 && experience_replay(i)# TODO: magic number
        end

        if (i-1) % opts.eval_interval == 0 # Simulate without noise and evaluate cost
            periodic_evaluation(i)
        end

    end
    println("Done. Minimum cost: $(minimum(cost[1:opts.eval_interval:end])), ($(minimum(cost)))")
    return cost, Θb, mem # Select the parameters with lowest cost
end


function update_actor!(Θ,αΘ,dΘ,dΘs,dΘs2,update_actor,opts)
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

function reduce_stepsize_periodic!(i,αΘ,αw,opts)
    if i % opts.stepreduce_interval != 0
        return αΘ,αw
    end
    αΘ  *= opts.stepreduce_factor
    αw  *= opts.stepreduce_factor
    αΘ,αw
end

function reduce_stepsize_divergence!(Θ,αΘ,αw,Θb)
    print_with_color(:orange,"Reducing stepsizes due to divergence\n")
    αΘ  /= 5
    αw  /= 5
    Θ[:] = deepcopy(Θb) # reset parameters
    αΘ,αw
end
