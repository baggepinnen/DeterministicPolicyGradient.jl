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
r(s,a) = (-(s)'Q1*(s) - a'Q2*a)[1]
L = lqr(ss(G),Q1,Q2)
μlqr(x) = -L*x


# Initialize solver options ==========================================
σβ      = 2
αΘ      = 0.005
αw      = 0.001
αv      = 0.001
αu      = 0.01
γ       = 0.999
τ       = 0.01
iters   = 10_000
rls_critic = true
λrls    = 0.99999
const opts = DPGopts(σβ,αΘ,αw,αv,αu,γ,τ,iters,m,rls_critic,λrls)


# Initialize functions      ==========================================
cp = linspace(-5,5,p)
cv = linspace(-5,5,p)
ca = linspace(-10,10,p)
grid1 = meshgrid(cp,cv)
grid2 = meshgrid2(cp,cv,ca)
const c1 = [grid1[1][:] grid1[2][:]]
const c2 = [grid2[1][:] grid2[2][:] grid2[3][:]]
P = size(c1,1)
P2 = size(c2,1)

function ϕ(s)
    a = exp(-1/(2*2)*sum((s'.-c1).^2,2))#exp(-1/2*(s.-c).^2)
    a ./= sum(a)
    a
    # [a; s[2]]
end

μ(s,Θ)          = Θ'ϕ(s)
∇μ(s)           = ϕ(s)
β(s,Θ,noise,i)  = Θ'ϕ(s) + noise[i]
ϕ(s,a,Θ)        = ∇μ(s)*(a-μ(s,Θ))
V(s,v)          = v'ϕ(s) # It's a good idea to bias V to some mean of the final landscape
Q(s,a,v,w,Θ)    = (ϕ(s,a,Θ)'w + V(s,v))[1]
simulate(Θ,x0, noise) = lsim(G, (i,s)->β(s,Θ,noise,i), t, x0)[3:4]
simulate(Θ,x0) = lsim(G, (i,s)->μ(s,Θ), t, x0)[3:4]
exploration(σβ) = filt(ones(5),[5],σβ*randn(T))
funs            = DPGfuns(μ,∇μ,β,ϕ,V,Q, simulate, exploration, r)

cost, Θ, w, v = dpg(opts, funs, x0)

@test minimum(cost) < 0.4cost[2]
