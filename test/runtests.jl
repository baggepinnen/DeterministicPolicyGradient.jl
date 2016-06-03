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


# Initialize solver options ==========================================
σβ                  = 2
αΘ                  = 0.01
αw                  = 0.01
αv                  = 0.01
αu                  = 0.01
γ                   = 0.999
τ                   = 0.01
iters               = 2_000
critic_update       = :rls
λrls                = 0.99999
stepreduce_interval = 5000
stepreduce_factor   = 0.995
hold_actor          = 100
opts = DPGopts(σβ,αΘ,αw,αv,αu,γ,τ,iters,m,critic_update,λrls,stepreduce_interval,stepreduce_factor,hold_actor)


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
Q(s,a,v,w,Θ,t)    = (ϕ(s,a,Θ)'w + V(s,v))[1]

function gradients(s1,s,a1,a,Θ,w,v,t)
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
w               = 0.001randn(P*m)
v               = 0.001randn(P)
initial_state   = DPGstate(Θ,w,v)

cost, Θ, w, v = dpg(opts, funs, initial_state, x0)
@test minimum(cost) < 0.4cost[2]

opts = DPGopts(σβ,αΘ,αw,αv,αu,γ,τ,iters,m,:kalman,λrls,stepreduce_interval,stepreduce_factor,hold_actor)
cost, Θ, w, v = dpg(opts, funs, initial_state, x0)
@test minimum(cost) < 0.4cost[2]

opts = DPGopts(σβ,αΘ,αw,αv,αu,γ,τ,iters,m,:gradient,λrls,stepreduce_interval,stepreduce_factor,hold_actor)
cost, Θ, w, v = dpg(opts, funs, initial_state, x0)
@test minimum(cost) < 0.5cost[2]
