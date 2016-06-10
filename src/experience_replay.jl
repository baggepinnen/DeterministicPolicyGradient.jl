r5(x) = round(x,5)
const b = Beta(1,5)

import Base.Collections: heapify, heapify!, heappush!, heappop!
import Base: isless, zero, push!, show, display, getindex, sort!

export Transition, ReplayMemory, sample!, push!, sort!

type Transition
    s::Vector{Float64}
    s1::Vector{Float64}
    a::Vector{Float64}
    r::Float64
    δ::Float64
end

display(t::Transition) = println("s: ", string(r5(t.s)), " s1: ", string(r5(t.s1)), " a: ", string(r5(t.a)), " r: ", string(r5(t.r)), " δ: ", string(r5(t.δ)))
show(t::Transition) = display(t)

isless(t1::Transition,t2::Transition) = isless(abs(t1.δ),abs(t2.δ))
zero(Transition) = Transition([0.],[0.],[0.],0.,-Inf)


type ReplayMemory
    mem::Vector{Transition}
    s::Int
    rs::Float64
    ReplayMemory(s::Int) = new(zeros(Transition,s),s, sum(1./(1:s)))
end

getindex(mem::ReplayMemory, ind) = mem.mem[n]
sort!(mem::ReplayMemory) = sort!(mem.mem,rev=true)

function display(mem::ReplayMemory)
    map(display,mem.mem)
    nothing
end

function push!(mem::ReplayMemory, t::Transition)
    heappush!(mem.mem, t, Base.Order.Reverse)
    pop!(mem.mem)
    nothing
end

function sample_greedy!(mem::ReplayMemory)
    # heappop!(mem.mem, Base.Order.Reverse)
    mem[1]
end

function sample_beta!(mem::ReplayMemory)
    s = mem.s
    rs = mem.rs
    ind = ceil(Int,s*rand(b))
    mem[ind]
end
