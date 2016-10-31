# workspace()
# gr()
using ControlSystems
using DeterministicPolicyGradient
using Plots
using TensorFlow
using ValueHistories
try; close("all");catch;end
issmall(x) = abs(x) < 1e-10
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
σβ                      = 0.2 # 0.01
αw                      = 0.0001 # 0.02

## Initialize solver options ===================================================
eval_interval           = 2
dpgopts = DPGopts(m,
αΘ                      = 0.0001, #0.05
γ                       = 0.99,
τ                       = 0.05,
iters                   = 250,
stepreduce_interval     = 100,
stepreduce_factor       = 0.99,
hold_actor              = 30,
experience_replay       = 200,
experience_ratio        = 10,
momentum                = 0.3,
rmsprop                 = true,
rmspropfactor           = 0.9,
eval_interval           = eval_interval,
divergence_threshold    = 1.8)
gradient_batch_size     = T
# Initialize functions =========================================================

const Q1        = [1,1]
const Q2        = 0.1*ones(m)
function r(s,a,i)
    local reward = -Q1⋅s.^2 - Q2⋅a.^2
    reward
end
Lq = lqr(ss(G),diagm(Q1),diagm(Q2))
μlqr(x) = -Lq*x


function μ(s,Θ,i)
    -Θ's
end
function ∇μ(s,i)
    -s
end
β(s,Θ,noise,i)  = μ(s,Θ,i) + noise[round(Int,i)]

# Batch norm stuff
N_tot = 0
running_sum = zeros(1,n)
running_ss = zeros(1,n)
running_mean() =  running_sum/N_tot
function running_std()
    local m = running_mean()
    sqrt((running_ss - 2*m.*running_sum)/N_tot + m.^2)
end

# Define network
session = Session(Graph())
a_ = placeholder(Float32, shape=[-1,m])
s_ = placeholder(Float32, shape=[-1,n])
y_ = placeholder(Float32, shape=[-1,1])
∇μi_ = placeholder(Float32, shape=[-1,m*n,m])
keep_prob = placeholder(Float32)
mean_ = placeholder(Float32, shape=[1,n])
std_ = placeholder(Float32, shape=[1,n])

seed = rand(1:1000)
srand(seed)

W1 = Variable(autoscale(n, 50), name="weights1")
W2 = Variable(autoscale(50, 50), name="weights2")
W2a= Variable(autoscale(m, 50), name="weights2a")
W3 = Variable(autoscale(50, 1), name="weights3")

B1 = Variable(0*rand(Float32,50), name="bias1")
B2 = Variable(0*rand(Float32,50), name="bias2")
B3 = Variable(-5*ones(Float32,1  ), name="bias3")

beta_bn = Variable(zeros(Float32,1,n),name="beta")
gamma_bn = Variable(ones(Float32,1,n),name="gamma")

# These lines define the network structure
l1 = ((s_-mean_).*gamma_bn/(std_+1e-6) + beta_bn)*W1 + B1 |> nn.relu
l1 = nn.dropout(l1, keep_prob)
l2 = l1*W2 + a_*W2a + B2 |> nn.relu # Include actions only in second layer
q  =          l2*W3 + B3
# q =         s_*W1  + a_*W2a + B3

srand(seed)
W1t = Variable(autoscale(n, 50), name="weights1t", trainable=false)
W2t = Variable(autoscale(50, 50), name="weights2t", trainable=false)
W2at= Variable(autoscale(m, 50), name="weights2at", trainable=false)
W3t = Variable(autoscale(50, 1), name="weights3t", trainable=false)

B1t = Variable(0*rand(Float32,50), name="bias1t", trainable=false)
B2t = Variable(0*rand(Float32,50), name="bias2t", trainable=false)
B3t = Variable(-5*ones(Float32,1  ), name="bias3t", trainable=false)

beta_bnt = Variable(zeros(Float32,1,n),name="betat", trainable=false)
gamma_bnt = Variable(ones(Float32,1,n),name="gammat", trainable=false)

# These lines define the network structure
l1t = ((s_-mean_).*gamma_bnt/(std_+1e-6) + beta_bnt)*W1t + B1t |> nn.relu
l1t = nn.dropout(l1t, keep_prob)
l2t = l1t*W2t + a_*W2at  + B2t |> nn.relu # Include actions only in second layer
qt  =          l2t*W3t   + B3t


Q(s1::Vector,a1::Vector,t)  = run(session, q,  Dict(s_ => s1', a_ => a1', keep_prob => one(Float32), mean_ => running_mean(), std_ => running_std()))[1]
Qt(s1::Vector,a1::Vector,t) = run(session, qt, Dict(s_ => s1', a_ => a1', keep_prob => one(Float32), mean_ => running_mean(), std_ => running_std()))[1]

∇aQ = TensorFlow.gradients(q,[a_])[1]
∇aQ = reshape(∇aQ,[-1,m,1])
dΘf = ⋅(∇μi_ , ∇aQ) # This weird operator is batch_matmul

sum_square = reduce_mean((y_ - q).^2)
weight_decay = 0.001*(reduce_sum(W1.^2) + reduce_sum(W2.^2) + reduce_sum(W2a.^2) + reduce_sum(W3.^2))
loss = sum_square #+ weight_decay
train_step = train.minimize(train.AdamOptimizer(αw), loss)
# optimizer = train.AdamOptimizer(αw)
# gvs = train.compute_gradients(optimizer,loss)
# capped_gvs = [(clip_by_value(grad, Float32(-0.01), Float32(0.01)), var) for (grad, var) in gvs]
# train_step = train.apply_gradients(optimizer,capped_gvs)

value_history = QHistory(Float32)
# push!(value_history,Float32(1.))

batch_size = 5
function train_critic(s,a,targets)
    l = size(s,1)
    mi = mean(s,1)
    si = std(s,1)

    global N_tot += l
    # N_tot = min(N_tot,10000)
    global running_sum += sum(s,1)
    global running_ss += sum(s.^2,1)

    # @show si, running_std()
    # @show mi, running_mean()

    # perm = randperm(l)
    # s = s[perm,:]
    # a = a[perm,:]
    # targets = targets[perm]

    steps = 1:batch_size:l-batch_size+1
    ts = zero(Float32)
    # ts = run(session, [sum_square, train_step], Dict(s_ => s, a_ => a, y_ => targets, keep_prob=>.95, mean_ => mi, std_ => si))[1]
    for i = steps
        ts += run(session, [sum_square, train_step], Dict(s_ => s[i:i+2,:], a_ => a[i:i+2,:], y_ => targets[i:i+2], keep_prob=>1., mean_ => mi, std_ => si))[1]
    end
    if steps[end] != l
        ts += run(session, [sum_square, train_step], Dict(s_ => s[steps[end]+1:end,:], a_ => a[steps[end]+1:end,:], y_ => targets[steps[end]+1:end], keep_prob=>1., mean_ => mi, std_ => si))[1]
    end
    push!(value_history,ts)
end

function update_tracking_networks()
    τ  = dpgopts.τ
    op = [
    assign(W1t,τ*W1   + (1-τ)*W1t),
    assign(W2t,τ*W2   + (1-τ)*W2t),
    assign(W2at,τ*W2a + (1-τ)*W2at),
    assign(W3t,τ*W3   + (1-τ)*W3t),
    assign(B1t,τ*B1   + (1-τ)*B1t),
    assign(B2t,τ*B2   + (1-τ)*B2t),
    assign(B3t,τ*B3   + (1-τ)*B3t),
    assign(gamma_bnt,τ*gamma_bn   + (1-τ)*gamma_bnt),
    assign(beta_bnt,τ*beta_bn   + (1-τ)*beta_bnt)]
    run(session,op)
end

function grad_theta(s,a,Θ,t)
    batch_size = size(s,1)
    ∇μi = zeros(batch_size, m*n, m)
    for i = 1:batch_size
        ∇μi[i,:,:]  = ∇μ(s[i,:],t[i])
    end
    dΘ  = sum(run(session, dΘf, Dict(s_ => s, a_ => a, ∇μi_ => ∇μi, keep_prob => one(Float32), mean_ => running_mean(), std_ => running_std())), 1)[:]
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
initial_state   = DPGstate(Θ)

pfig = plot(layout=3)
function progressplot(Θ,i,s,u, cost, rollout)

        plot!(pfig, s, lab="It: $i c: $(cost[i] |> r3)", c=[:blue :red], subplot=1)
        Qplot!(rollout, Q, μ, Θ, dpgopts.γ, subplot=2)
        plot!(value_history, yscale=:log10, subplot=3)
        # update_plot!(pfig[1], attribute=:linecolor, max_history=20)
        # update_plot!(pfig[2])
        # update_plot!(pfig[3], max_history=1)
        gui(pfig)
end

## Solve DPG ===================================================================
run(session, initialize_all_variables())
cost, Θ, mem = dpg(dpgopts, funs, initial_state, x0, progressplot)
# plot(cost,c=:red)
# plot!(1:eval_interval:length(cost),cost[1:eval_interval:end],linewidth=3)
# gui()

# Test =========================================================================


plot(lsim(G, (i,s)->μ(s,Θ,i), linspace(0,T,200),x0)[3], lab="After")
plot!(lsim(G, (i,s)->μlqr(s), linspace(0,T,200),x0)[3], lab="LQR")
gui()

if true
    s1vec = linspace(-1,1,100)
    s2vec =linspace(-1,1,100)
    uvec = linspace(-5,5,100)

    control_map = Float64[μ([s1,s2],Θ,1)[1] for s2 in s2vec, s1 in s1vec]
    heatmap(s1vec,s2vec,control_map,title="Control map",ylabel="Position 2",xlabel="Position 1");
    gui()

end
