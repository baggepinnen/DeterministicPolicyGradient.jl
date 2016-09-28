using DeterministicPolicyGradient, ControlSystems
using Base.Test


const G = tf(0.1,[1,0.01,0.1])

const T = 20
const t = 1:T
const m = 1
const n = 2
const p = 5
x0 = [0,1.]

# Cost function             ==========================================
const Q1 = diagm([5,1])
const Q2 = 1eye(m)
# r(s,a) = -sum(abs(Q1*s)) - sum(abs(Q2*a))
r(s,a,t) = (-(s)'Q1*(s) - a'Q2*a)[1]
L = lqr(ss(G),Q1,Q2)
μlqr(x) = -L*x


# Initialize functions      ==========================================
cp = linspace(-5,5,p)
cv = linspace(-5,5,p)
grid1 = meshgrid(cp,cv)
const c1 = [grid1[1][:] grid1[2][:]]
P = size(c1,1)

function ϕ(s)
    a = exp(-1/(2*2)*sum((s'.-c1).^2,2))#exp(-1/2*(s.-c).^2)
    a ./= sum(a)
    a
    # [a; s[2]]
end

μ(s,Θ,t)          = Θ'ϕ(s)
∇μ(s)           = ϕ(s)
β(s,Θ,noise,i)  = Θ'ϕ(s) + noise[i]
ϕ(s,a,Θ)        = ∇μ(s)*(a-μ(s,Θ,t))
V(s,v)          = v'ϕ(s) # It's a good idea to bias V to some mean of the final landscape
Q(s,a,v,w,Θ,t,c)    = (ϕ(s,a,Θ)'w + V(s,v))[1]

function gradients(s1,s,a1,a,Θ,w,v,t,c)
    ∇μ = ϕ(s)
    ∇aQ = ∇μ'w
    ∇wQ = ∇μ*(a-Θ'∇μ)
    ∇vQ = ∇μ
    ∇aQ, ∇wQ, ∇vQ, ∇μ
end

simulate(Θ,x0, noise) = lsim(G, (t,s)->β(s,Θ,noise,t), t, x0)[3:4]
simulate(Θ,x0) = lsim(G, (t,s)->μ(s,Θ,t), t, x0)[3:4]
exploration(σβ) = filt(ones(5),[5],σβ*randn(T))
funs            = DPGfuns(μ,Q, gradients, simulate, exploration, r)

Θ               = zeros(P*m) # Weights
w               = 0.00001randn(P*m)
v               = 0.00001randn(P)
initial_state   = DPGstate(Θ,w,v)


opts = DPGopts(m,
σβ                  = 1,
αΘ                  = 0.001,
γ                   = 0.9999,
τ                   = 0.01,
iters               = 250,
critic_update       = :rls,
λrls                = 0.99999,
stepreduce_interval = 1,
stepreduce_factor   = 0.995,
hold_actor          = 10,
experience_replay   = 0,
experience_ratio    = 10,
momentum            = 0.8,
rmsprop             = true,
rmspropfactor       = 0.9,
critic_update_interval = 3,
eval_interval       = 10,
divergence_threshold  = 1.5)

cost, Θ, w, v, mem = dpg(opts, funs, initial_state, x0, c1)
@test minimum(cost) < 0.4cost[2]

opts = DPGopts(m,
σβ                  = 2,
αΘ                  = 0.001,
γ                   = 0.999,
τ                   = 0.01,
iters               = 250,
critic_update       = :kalman,
λrls                = 0.99999,
stepreduce_interval = 1,
stepreduce_factor   = 0.995,
hold_actor          = 10,
experience_replay   = 0,
experience_ratio    = 10,
momentum            = 0.8,
rmsprop             = true,
rmspropfactor       = 0.9,
critic_update_interval = 1,
eval_interval       = 10,
divergence_threshold  = 1.5)

cost, Θ, w, v, mem = dpg(opts, funs, initial_state, x0, c1)
@test minimum(cost) < 0.4cost[2]

opts = DPGopts(m,
σβ                  = 2,
αΘ                  = 0.001,
αw                  = 0.01,
αv                  = 0.01,
γ                   = 0.99,
τ                   = 0.01,
iters               = 3000,
critic_update       = :gradient,
stepreduce_interval = 10,
stepreduce_factor   = 0.995,
hold_actor          = 500,
experience_replay   = 0,
experience_ratio    = 1,
momentum            = 0.8,
rmsprop             = true,
rmspropfactor       = 0.9,
eval_interval       = 100,
divergence_threshold  = 1.5)
# Det var något vajsing med att δ aldrig blev uträknad vid gradient training
cost, Θ, w, v, mem = dpg(opts, funs, initial_state, x0, c1)
@test minimum(cost) < 0.5cost[2]

mem



## Test experience replay
n = 2
s = randn(n)
s1 = randn(n)
a = randn(n)
reward = randn()
δ1 = 1.
δ2 = 2.
δ3 = 3.

t1 = Transition(s,s1,a,reward,δ1,1,1)
t2 = Transition(s,s1,a,reward,δ2,2,2)
t3 = Transition(s,s1,a,reward,δ3,3,3)

@test t1 < t2

N = 3
mem = SortedReplayMemory(N)

push!(mem,t2)
push!(mem,t1)
push!(mem,t3)
@test length(mem.mem) == N
@test Collections.isheap(mem.mem, Base.Order.Reverse)
sort!(mem)
@test Collections.isheap(mem.mem, Base.Order.Reverse)

tt = sample_greedy!(mem)
@test tt.δ == 3.
