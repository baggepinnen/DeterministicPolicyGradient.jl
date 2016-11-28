gr()
using ControlSystems
using DeterministicPolicyGradient
using Plots
using TensorFlow
using ValueHistories
try; close("all");catch;end
r3(x) = round(x,3)
default(show=false, colorbar=false, size=(1400,1100))


## Setup =======================================================================
const G = tf(0.1,[1,0.01,0.1])
const T = 20
const time = 1:T
const m = 1
const n = 2
x0 = [0,1.]

# Setup DPG ====================================================================
const m = 1
σβ      = 0.2 # 0.01
αw      = 0.04 # 0.02

## Initialize solver options ===================================================
eval_interval        = 5
dpgopts              = DPGopts(m,
αΘ                   = 0.001, #0.05
γ                    = 0.999,
τ                    = 0.1,
iters                = 200,
stepreduce_interval  = 3,
stepreduce_factor    = 0.99,
hold_actor           = 20,
experience_replay    = 1000,
experience_ratio     = 30,
momentum             = 0.1,
rmsprop              = true,
rmspropfactor        = 0.9,
eval_interval        = eval_interval,
divergence_threshold = 1.8)
gradient_batch_size  = T
# Initialize functions       =========================================================

const Q1 = [1,1]
const Q2 = 0.1*ones(m)
r(s,a,i) = -Q1⋅s.^2 - Q2⋅a.^2

Lq       = lqr(ss(G),diagm(Q1),diagm(Q2))
μlqr(x)  = -Lq*x


function μ(s,Θ,i)
    -Θ's
end
function ∇μ(s,i)
    -s
end
β(s,Θ,noise,i)  = μ(s,Θ,i) + noise[round(Int,i)]

# Define network
session = Session(Graph())
a_      = placeholder(Float32, shape=[-1,m])
s_      = placeholder(Float32, shape=[-1,n])
y_      = placeholder(Float32, shape=[-1,1])
∇μi_    = placeholder(Float32, shape=[-1,m*n,m])

sa = concat(1, [s_,a_]) # Python indexing of dims

seed = rand(1:1000)
srand(seed)
W = Variable((0.001*randn(Float32,n+m,n+m)), name="Qweight")
C = Variable(0.001*randn(Float32,n+m, 1), name="Lweight")
B = Variable(0*rand(Float32,1), name="bias")

q = reduce_sum((sa*W).*sa,reduction_indices=[2]) + sa*C + B

srand(seed)
Wt = Variable((0.001*randn(Float32,n+m,n+m)), name="Qweightt", trainable=false)
Ct = Variable(0.001*randn(Float32,n+m, 1), name="Lweightt", trainable=false)
Bt = Variable(0*rand(Float32,1), name="biast", trainable=false)

qt = reduce_sum((sa*Wt).*sa,reduction_indices=[2]) + sa*Ct + Bt

Q(s1::Vector,a1::Vector,t)  = run(session, q,  Dict(s_ => s1', a_ => a1'))[1]
Qt(s1::Vector,a1::Vector,t) = run(session, qt, Dict(s_ => s1', a_ => a1'))[1]

∇aQ = TensorFlow.gradients(q,[a_])[1]
∇aQ = reshape(∇aQ,[-1,m,1])
dΘf = batch_matmul(∇μi_ , ∇aQ)

loss = reduce_mean((y_ - q).^2)
loss = loss
train_step = train.minimize(train.AdamOptimizer(αw), loss)

value_history = QHistory(Float32)

function train_critic(s,a,targets)
    l = size(s,1)
    ts = run(session, [loss, train_step], Dict(s_ => s, a_ => a, y_ => reshape(targets,l,1)))[1]

    push!(value_history,ts)
end

function update_tracking_networks()
    τ  = dpgopts.τ
    op = [
    assign(Wt,τ*W + (1-τ)*Wt),
    assign(Ct,τ*C + (1-τ)*Ct),
    assign(Bt,τ*B + (1-τ)*Bt)]
    run(session,op)
end

function grad_theta(s,a,Θ,t)
    batch_size = size(s,1)
    ∇μi = zeros(batch_size, m*n, m)
    for i = 1:batch_size
        ∇μi[i,:,:]  = ∇μ(s[i,:],t[i])
    end
    dΘ  = sum(run(session, dΘf, Dict(s_ => s, a_ => a, ∇μi_ => ∇μi)), 1)[:]
end

exploration(σβ) = σβ*randn(T)

function simulate(Θ,x0, noise = false)
    if noise
        e = exploration(σβ)
        lsim(G, (t,s)->β(s,Θ,e,t), time, x0)[3:4]
    else
        lsim(G, (t,s)->μ(s,Θ,t), time, x0)[3:4]
    end
end

funs            = DPGfuns(μ,Q,Qt, grad_theta, simulate, r, train_critic, update_tracking_networks)
Θ               = 0.001randn(n)


pfig = plot(layout=4)
function progressplot(Θ,i,s,u, cost, rollout)

        plot!(pfig, s, lab=["It: $i c: $(cost[i] |> r3)" ""], c=[:blue :red], subplot=1)
        Qplot!(rollout, Q, μ, Θ, dpgopts.γ, subplot=2)
        plot!(value_history, yscale=:log10, subplot=3)
        plot!(cost[1:i], title="Cost", subplot=4)
        update_plot!(pfig[1], attribute=:linecolor, max_history=40)
        update_plot!(pfig[2])
        update_plot!(pfig[3], max_history=1)
        update_plot!(pfig[4], max_history=1)
        gui(pfig)
end

## Solve DPG ===================================================================
run(session, initialize_all_variables())
cost, Θ, mem = dpg(dpgopts, funs, Θ, x0, progressplot)
# plot(cost,c=:red)
# scatter!(1:eval_interval:length(cost),cost[1:eval_interval:end],linewidth=3)
# gui()

# Test =========================================================================


plot(lsim(G, (i,s)->μ(s,Θ,i), linspace(0,T,200),x0)[3], lab="After", reuse=false)
plot!(lsim(G, (i,s)->μlqr(s), linspace(0,T,200),x0)[3], lab="LQR")
gui()

if false
    s1vec = linspace(-1,1,100)
    s2vec = linspace(-1,1,100)
    uvec  = linspace(-5,5,100)

    control_map = Float64[μ([s1,s2],Θ,1)[1] for s2 in s2vec, s1 in s1vec]
    heatmap(s1vec,s2vec,control_map,title="Control map",ylabel="Position 2",xlabel="Position 1");
    gui()

end
